/*****************************************************************************
 
 Copyright (c) 2009-2011, Sascha Lange
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

/*
 *  Part of LayerTypes.cpp
 *  that was created by Sascha Lange on 30.04.09.
 */


#include "FullyConnectedLayer.h"
#include "npp2.h"
#include <cstdlib>
#include <iostream>
#include <functions.h>
#include "PatternSet.h"

#if  (defined(__APPLE_CPP__) || defined(__APPLE_CC__) || defined(__MACOS_CLASSIC__))
#include <veclib/cblas.h>
#else
#include <cblas.h>
#endif

#include "Registry.h"
#include <cassert>

using namespace std;
using namespace NPP2;


FullyConnectedLayer::FullyConnectedLayerArguments::FullyConnectedLayerArguments( int numUnits)
: BasicLayerArguments(numUnits, 1)
{ identifer = "FullyConnectedLayer"; }

FullyConnectedLayer::FullyConnectedLayerArguments::FullyConnectedLayerArguments( int numCols, int numRows)
: BasicLayerArguments(numCols, numRows)
{ identifer = "FullyConnectedLayer"; }


FullyConnectedLayer::FullyConnectedLayer(Net* net, int layerId, int firstUnitId, int unitsPerRow, int numRows, int numCopies)
: BasicLayerType(net, layerId, firstUnitId, unitsPerRow, numRows, numCopies), weights(0), variables(0), previousDim(0)
{
  identifer = "FullyConnectedLayer";
}

FullyConnectedLayer::FullyConnectedLayer() 
: BasicLayerType(0, 0, 0, 0, 0, 0), weights(0), dEdw(0), delta(0), variables(0), previousDim(0) 
{
  identifer = "FullyConnectedLayer";
}

FullyConnectedLayer::FullyConnectedLayer(Net* net, int layerId, const LayerArguments* args)
: BasicLayerType(net, layerId, args), weights(0), variables(0), previousDim(0)
{
  identifer = "FullyConnectedLayer";
}


LayerArguments* FullyConnectedLayer::getArguments() const
{
  return new FullyConnectedLayerArguments( numCols, numRows );
}

FullyConnectedLayer::~FullyConnectedLayer() 
{
  if (weights) {
    delete [] weights;
    delete [] dEdw;
    delete [] delta;
  }
  if (variables) {
    delete [] variables;
  }
  if (updateFunction) {
    delete updateFunction;
  }
}

void FullyConnectedLayer::connectLayer(const BasicLayerType* previousLayer)
{
  this->previousDim = previousLayer->numUnits;
  this->numWeights = (previousDim+1) * numUnits;
  
  weights = new FTYPE[(previousDim+1) * numUnits];
  delta = new FTYPE[(previousDim+1) * numUnits];
  dEdw = new FTYPE[(previousDim+1) * numUnits * (numCopies+1)];  // n-copies, used by the n-threads to accumulate deriv. for patterns
  
  memset(delta, 0, sizeof(FTYPE) * (previousDim+1)*numUnits);
  memset(dEdw, 0, sizeof(FTYPE) * (previousDim+1)*numUnits*(numCopies+1));
  
  if (updateFunction) {
    setUpdateFunction(updateFunction);  // re-set the updateFunction in order to create and initialize the variable-vector
  }
}

void FullyConnectedLayer::setUpdateFunction(const UpdateFunction* updateFunction)
{
  const UpdateFunction* oldf = this->updateFunction;
  this->updateFunction = updateFunction->clone();
  
  if (oldf) {
    delete oldf;
  }
  
  if (weights) {
    if (variables) {
      delete [] variables;
    }
    int numVariables = this->updateFunction->getNumVariables();
    variables = new FTYPE[(previousDim+1) * numUnits * numVariables];
    for (int i=0; i < (previousDim+1) * numUnits; i++) {
      this->updateFunction->initVariables(&variables[i*numVariables]);
    }
  }
}



void FullyConnectedLayer::initWeights(int mode, FTYPE range)
{
  if(mode == 0){
    for(int i=0; i < numWeights; i++) {
      weights[i] = (FTYPE)((2.0 * range * drand48()) -range);
    }
  }
  else if(mode == 1){ /* all biases = 0.0 */
    /*
     for(i=topo_data.in_count+1;i<=topo_data.unit_count;i++){ 
     if((wptr=unit[i].weights)!=NULL){
     wptr->value = (FTYPE) 0;
     wptr=wptr->next;
     for (; wptr!=NULL; wptr=wptr->next)
     wptr->value = (FTYPE)((2.0 * range * drand48()) -range);
     }
     }
     */
  }
}

void FullyConnectedLayer::forwardPass(FTYPE *input, int copy)
{
  int pos = copy*(numUnits+1);
  cblas_dgemv (CblasRowMajor, CblasNoTrans, numUnits, previousDim+1,    // M = Ausgabevektor mit netins. N = Eingabevektor mit Ausgabe der vorherigen Schicht (+1 Bias-Neuron)
               1., weights, previousDim+1, input, 1, 0., &netin[pos+1], 1); // lda ist bei row-major die Spaltenanzahl previousDim+1
  for (int i=1; i <= numUnits; i++) {
    out[pos+i] = act_f(netin[pos+i]);
  }
}

void FullyConnectedLayer::backwardPass(FTYPE *dedout, int copy)
{
  int pos = copy*(numUnits+1);
  int posWeightMatrices = copy*(previousDim+1) * numUnits; // this points to the correct copy of the weights matrices and the corresponding derivatives
  for (int i=1; i <= numUnits; i++) {
    dEdnet[pos+i] = dEdo[pos+i] * deriv_f(out[pos+i], netin[pos+i]);
    dEdo[pos+i] = (FTYPE) 0;
  }
  if (getLayerType() == INPUT_LAYER) {
    memcpy(dedout, &(dEdnet[pos+1]), sizeof (FTYPE) * numUnits);
    return; // ready. Otherwise calc derivs for weights and output of previous layer.
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, numUnits, previousDim+1, 1, 1., &dEdnet[pos+1], 1, 
              &net->layers[layerId-1]->out[copy*(previousDim+1)], previousDim+1, 1.,  // -> 1 in order to sum up over the patterns!
              &dEdw[posWeightMatrices], previousDim+1); // sum up in correct copy of dEdw
  
  cblas_dgemv(CblasRowMajor, CblasTrans, numUnits, previousDim, 1., weights+1, previousDim+1, &dEdnet[pos+1], 1, 0., dedout, 1);  // skip bias
}

void FullyConnectedLayer::updateWeights(int numThreads)
{
  if (!updateFunction || !weights || !variables) {
    cerr << "Layer " << layerId << " not correctly initialized." << endl;
    exit(1);
  }
  
  if (numThreads > 1) {
    for (int i=1; i <= numThreads; i++) {
      cblas_daxpy((previousDim+1)*numUnits, 1., &dEdw[(previousDim+1)*numUnits*i], 1, dEdw, 1); // sum the partial sums of the derivatives
      cblas_dscal((previousDim+1)*numUnits, 0., &dEdw[(previousDim+1)*numUnits*i], 1);          // and set the copies back to zero (for the next iteration)
    }
  }
  
  int numVariables = updateFunction->getNumVariables();
  for (int to=0; to < numUnits; to++) {
    for (int from=0; from < previousDim+1; from++) {
      int pos = (previousDim+1)*to+from;
      //double save = weights[pos];
      (*updateFunction) (&weights[pos], &delta[pos], &dEdw[pos], &variables[pos*numVariables]); // now calculte the weight change and apply it
    }
  }  
}


void FullyConnectedLayer::copyWeights(const BasicLayerType* layer)
{
  const FullyConnectedLayer* flayer;
  try {
    flayer = dynamic_cast<const FullyConnectedLayer*> (layer);
  } catch (exception& e) {
    cerr << "Tried to copy weights from a layer of a different type." << endl;
    exit(1);
  }
  if (!weights) {
    connectLayer(net->layers[layerId-1]);
  }
  if (flayer->numUnits != numUnits) {
    cerr << "The target layer has not as many neurons as the source layer." << endl;
    exit(1);
  }
  if (flayer->previousDim != previousDim) {
    cerr << "The target's layer previous layer has not as many neurons as the source layer's previous layer." << endl;
    exit(1);
  }
  assert (numWeights == flayer->numWeights);
  
  memcpy(weights, flayer->weights, flayer->numWeights*sizeof(FTYPE));
}


// input and output

void BasicLayerType::writeToStream(std::ostream& out) const
{
  out << layerId << " " << identifer << endl << numUnits << " " << firstUnitId << " " << (firstUnitId+numUnits-1) << endl;
  out << numRows << " " << numCols << " " << numCopies << endl;
  out << actId << " " << endl;
}

void BasicLayerType::readFromStream(std::istream& in)
{
  int lastUnitId;
  in >> numUnits >> firstUnitId >> lastUnitId;
  assert(lastUnitId == firstUnitId + numUnits-1);
  in >> numRows >> numCols >> numCopies >> actId;
  assert (numRows*numCols == numUnits);
  
  BasicLayerType::initLayer();
}

void FullyConnectedLayer::writeToStream(std::ostream& out) const
{
  BasicLayerType::writeToStream(out);
  
  if (getLayerType() == INPUT_LAYER) return; // no weights in input layer.
  
  int size = (previousDim+1) * numUnits;
  out << previousDim << " " << numUnits << " " << size << endl;
  for (int i=0; i < numUnits; i++) {
    for (int j=0; j < (previousDim+1); j++) { // inklusive Biasneuron (ID: 0)
      out << weights[i*(previousDim+1) + j] << " ";
    }
    out << endl;
  }
}

void FullyConnectedLayer::readFromStream(std::istream& in)
{
  BasicLayerType::readFromStream(in);
  
  if (getLayerType() == INPUT_LAYER) return;
  
  int size;
  in >> previousDim >> numUnits >> size;
  assert ((previousDim+1) * numUnits == size);
  
  connectLayer(net->layers[layerId-1]);
  
  for (int i=0; i < numUnits; i++) {
    for (int j=0; j < (previousDim+1); j++) {
      in >> weights[i*(previousDim+1) +j];
    }
  }
}

REGISTER_LAYERTYPE(FullyConnectedLayer, "Standard layer type where each neuron is connected to all neurons of the preceeding layer.")




