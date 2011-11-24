/*****************************************************************************
 
 Copyright (c) 2008-2011, Sascha Lange, 
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without 
 modification, are permitted provided that the following conditions are met:
 
 - Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 
 - Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 
 - Neither the name of Sascha Lange Software nor the names of its 
 contributors may be used to endorse or promote products derived from this
 software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 POSSIBILITY OF SUCH DAMAGE.
 
 ****************************************************************************/


/*  DeepAutoEncoder.cpp
 *  Created by Sascha Lange on 03.10.08.
 */

#include "BasicLayerTypes.h"
#include "DeepAutoEncoder.h"
#include "npp2.h"
#include "PatternSet.h"
#include <exception>
#include <map>
#include <cassert>
#include <fstream>

using namespace std;
using namespace NPP2;


DeepAutoEncoder::DeepAutoEncoder(Net* net, bool own) : own(own)
{
  fullNet = net;
}


DeepAutoEncoder::~DeepAutoEncoder()
{
  if (own) delete fullNet;
}

void DeepAutoEncoder::pretrain(const PatternSet& patterns, const std::vector<CascadeParams>& params)
{
  pretrain(patterns, 1, params);
}


// training procedure that does not change the original net's structure
void DeepAutoEncoder::pretrain(const PatternSet& patterns, int numMiniBatches, const std::vector<CascadeParams>& params)
{  
  int numCascades = (fullNet->topoData.layerCount-1) / 2;  // determins the number of layer-wise pretraining procedures (cascades) to pretrain each layer. e.g. for a 3-layer network only one cascade
  
  
  // now create a pattern set with lists appropriate for holding the 
  // "local" training pattern (that are the activations of the previous layer)
  PatternSet localPattern; 
  
  localPattern.pattern_count = patterns.pattern_count;
  localPattern.input = new FTYPE*[patterns.pattern_count]; // we only need input 
  localPattern.input_count = patterns.input_count; 
  
  for (int i=0; i < patterns.pattern_count; i++) {
    localPattern.input[i] = new FTYPE[patterns.input_count]; // matches input layer
    for (int j=0; j < patterns.input_count; j++) {
      localPattern.input[i][j] = patterns.input[i][j];
    }
  }
  
  char buf[1024]; // buffer for creating filenames
  
  for (int c=0; c < numCascades; c++) { // works from outermost to innermost layer
    cout << "Starting cascade " << c << endl << flush;
    
    Net net;     // create new shallow autoencoder

    vector<LayerArguments*> netSpecification;
    netSpecification.push_back(fullNet->layers[c]->getArguments());  // layer in encoder part
    netSpecification.push_back(fullNet->layers[c+1]->getArguments());// subsequent layer will become the "code - layer"
    netSpecification.push_back(fullNet->layers[fullNet->topoData.layerCount-1-c]->getArguments()); // layer in decoder part corresponding to layer from encoder part

    net.createLayers(netSpecification, fullNet->numCopies, true);   // create layers of specified sizes
    
    net.layers[1]->copyWeights(fullNet->layers[c+1]);   // this call is not necessary for fully connected layers (will call initWeights in a second), but perhaps for other types such as IndividuallyConnectedLayers.
    net.layers[2]->copyWeights(fullNet->layers[fullNet->topoData.layerCount-1-c]); 

    
    FTYPE uparams[MAX_PARAMS] = { // either use parameters as specified or standard values (if no parameters specified for this cascade
      (int)params.size() > c ? (params[c].deltaStart > 0. ? params[c].deltaStart : params[c].deltaMax / 10.) : 0.01, 
      (int)params.size() > c ? params[c].deltaMax : .1, 
      (int)params.size() > c ? params[c].decay > 0. ? params[c].decay : params[c].deltaMax / 100. : .001,  0., 0., 0., 0., 0., 0., 0.};
    net.setUpdateFunc(0, uparams);  // RPROP
    net.initWeights(0,.5); // 1. / net.layers[0]->numUnits);

    
    assert(net.layers[0]->numUnits   == fullNet->layers[c]->numUnits);
    assert(net.layers[1]->numWeights == fullNet->layers[c+1]->numWeights);
    assert(net.layers[2]->numWeights == fullNet->layers[fullNet->topoData.layerCount-1-c]->numWeights);

    sprintf(buf, "pretrain_%d_init.net", c);
    net.saveNet(buf);   // save network, mainly for debugging purposes
    
    // //////////////// actual training of subnet //////////////////////
    double tss=0.;
    int epochs = (int)params.size() > c ? params[c].epochs : 50;
    assert (net.topoData.inCount == localPattern.input_count);
    assert (net.topoData.outCount == localPattern.input_count);

    for (int epoch=0; epoch < epochs; epoch++) {
      tss = net.train(&localPattern, net.numCopies, true, new SquaredError(), numMiniBatches);       // train for one epoch using as many threads as possible
      if (epoch % ((epochs / 10) < 1 ? 1 : epochs / 10) == 0) {
        cout << "Epoch " << epoch << " ss: " << (tss/patterns.pattern_count) << endl << flush;
      }
      if ((tss/ (localPattern.pattern_count)) < 0.01) { break; } // "early" stopping when finished training
    }
    cout << "Cascade " << c << " FINAL_TSS: " << tss << endl;
    sprintf(buf, "pretrain_%d_trained.net", c);
    net.saveNet(buf);
    // /////////////////////////////////////////////////////////////////
    
    
    // net.clearDerivatives();
    int newSize = net.layers[1]->numUnits;
    for (int p=0; p < localPattern.pattern_count; p++) { // propagte every pattern through the network and create a new training pattern for the next layer by storing activations of the inner hidden layer. The next inner layers will be trained on reproducing these exact activations. 
      net.forwardPass(localPattern.input[p], net.outVec);
      delete [] localPattern.input[p];   // delete old input pattern
      localPattern.input[p] = new double[newSize]; // create new with appropriate size
      for (int neuron=0; neuron < newSize; neuron++) {
        assert(net.layers[1]->out[0] == 1.);
        localPattern.input[p][neuron] = net.layers[1]->out[neuron+1];
      }
    }
    localPattern.input_count = newSize;
    
    // now copy back the pre-trained weights to the appropriate places
    fullNet->layers[c+1]->copyWeights(net.layers[1]); 
    fullNet->layers[fullNet->topoData.layerCount-1-c]->copyWeights(net.layers[2]);    
  }
}


  
double DeepAutoEncoder::train(const PatternSet& pattern, int numEpochs, ostream& out, const PatternSet* testPattern, int offset, int numBatches)
{
  double tss = 0.0;
  if (numEpochs == 0) {
    numEpochs = 1; ///< \todo : in this case train until error is "very small"
  }
  //fullNet->clear_derivatives();
    
  for (int epoch=0; epoch < numEpochs; epoch++) {
    
    if (testPattern && epoch % 10 == 0) {
      Error testerr = fullNet->test(testPattern, fullNet->numCopies, true);
      out << "MSE_TEST in epoch " << epoch + offset << ": " << testerr.regrError / (testPattern->pattern_count * fullNet->topoData.outCount) << endl;
    }
    
    tss = fullNet->train(&pattern, fullNet->numCopies, true,  new SquaredError(), numBatches);
    out << "MSE in epoch " << epoch + offset << ": " << tss / (pattern.pattern_count * fullNet->topoData.outCount) << endl; 
  }
  return tss;
}


Net* DeepAutoEncoder::deriveEncoderNet() const 
{ 
  vector<LayerArguments*> netSpecification;
  int numLayers = (fullNet->topoData.layerCount+1) / 2;
  
  for (int i=0; i < numLayers; i++) {
    netSpecification.push_back(fullNet->layers[i]->getArguments());
  }
 
  Net* net = new Net();
  net->createLayers(netSpecification, fullNet->numCopies, true);  
  
  for (int i=1; i < net->topoData.layerCount; i++) {
    net->layers[i]->copyWeights(fullNet->layers[i]);  
  }

  net->setUpdateFunc(0, fullNet->updateParams);  // TODO
  
  return net;
}
  



