/*****************************************************************************
 
 Copyright (c) 2009-2011, Sascha Lange, 
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

/*  Created by Sascha Lange on 30.04.09.
 *  Last modified: 17.11.2011
 */

#include "AdvancedLayerTypes.h"
#include "FullyConnectedLayer.h"
#include "npp2.h"
#include <cstdlib>
#include <iostream>
#include "functions.h"
#include "PatternSet.h"
#if  (defined(__APPLE_CPP__) || defined(__APPLE_CC__) || defined(__MACOS_CLASSIC__))
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <map>
#include <cassert>


using namespace NPP2;
using namespace std;


#ifdef __APPLE__
#pragma mark -
#pragma mark Layer Argument Classes
#endif

MultimodalCrossEntropyOutputLayer::MultimodalCrossEntropyOutputLayerArguments::MultimodalCrossEntropyOutputLayerArguments(int numUnits)
  : FullyConnectedLayerArguments(numUnits, 1)
{ identifer = "MultimodalCrossEntropyOutputLayer"; }

MultimodalCrossEntropyOutputLayer::MultimodalCrossEntropyOutputLayerArguments::MultimodalCrossEntropyOutputLayerArguments(int numCols, int numRows)
  : FullyConnectedLayerArguments(numCols, numRows)
{ identifer = "MultimodalCrossEntropyOutputLayer"; }



IndividuallyConnectedLayer::IndividuallyConnectedLayerArguments::IndividuallyConnectedLayerArguments(int numCols, int numRows)
  : BasicLayerArguments(numCols, numRows)
{ identifer = "IndividuallyConnectedLayer"; }



ConvolutionLayer::ConvolutionLayerArguments::ConvolutionLayerArguments(int numCols, int numRows, int numKernels, int kernelSize, int stepsize, bool shareWeights, bool overBoundary, bool sum, bool shareBias, bool interleave)
  : IndividuallyConnectedLayerArguments(numCols, numRows), numKernels(numKernels), kernelSize(kernelSize), stepsize(stepsize), shareWeights(shareWeights), overBoundary(overBoundary), sum(sum), shareBias(shareBias), interleave(interleave)
{ identifer = "ConvolutionLayer"; }

ConvolutionLayer::ConvolutionLayerArguments::ConvolutionLayerArguments( int numKernels, int kernelSize, int stepsize, bool shareWeights, bool overBoundary, bool sum, bool shareBias, bool interleave)
  : IndividuallyConnectedLayerArguments( 0, 0), numKernels(numKernels), kernelSize(kernelSize), stepsize(stepsize), shareWeights(shareWeights), overBoundary(overBoundary), sum(sum), shareBias(shareBias), interleave(interleave)
{ identifer = "ConvolutionLayer"; }



InvertedLayer::InvertedLayerArguments::InvertedLayerArguments(int numCols, int numRows)
  : IndividuallyConnectedLayerArguments( numCols, numRows)
{ identifer = "InvertedLayer"; }




#ifdef __APPLE__
#pragma mark -
#pragma mark Multimodal Cross Entropy Output Layer
#endif

MultimodalCrossEntropyOutputLayer::MultimodalCrossEntropyOutputLayer() 
: FullyConnectedLayer() 
{
  identifer = "MultimodalCrossEntropyOutputLayer";
}

MultimodalCrossEntropyOutputLayer::MultimodalCrossEntropyOutputLayer(Net* net, int layerId, const LayerArguments* args) 
: FullyConnectedLayer(net, layerId, args) 
{
  identifer = "MultimodalCrossEntropyOutputLayer";
}


MultimodalCrossEntropyOutputLayer::MultimodalCrossEntropyOutputLayer(Net* net, int layerId, int firstUnitId, int unitsPerRow, int numRows, int numCopies)
: FullyConnectedLayer(net, layerId, firstUnitId, unitsPerRow, numRows, numCopies)
{
  identifer = "MultimodalCrossEntropyOutputLayer";
}


MultimodalCrossEntropyOutputLayer::~MultimodalCrossEntropyOutputLayer() 
{}

LayerArguments* MultimodalCrossEntropyOutputLayer::getArguments() const
{
  return new MultimodalCrossEntropyOutputLayerArguments(numCols, numRows );
}

void MultimodalCrossEntropyOutputLayer::forwardPass(FTYPE *input, int copy)
{
  // procedure:
  // first, calculate the net inputs to all neurons
  // then, sum up all the activations of all neurons
  // finally, use the sum to "weight" the output of each individual neuron
  
  int pos = copy*(numUnits+1);
  double sum = 0.;
  
  // First: caclulate netinputs (using a BLAS matrix-vector operation)
  cblas_dgemv( CblasRowMajor,// matrix comes in row-major encoding
               CblasNoTrans, // do not transpose 
               numUnits,     // M = dim of output vector (will hold net inputs)
               previousDim+1,// N = dim of input (dim of previous layer +1 Bias-N.)
               1., 
               weights,      // weight matrix
               previousDim+1,// lda is-in case of row-major-the number of columns  
               input,        // input to this layer == output of previous layer
               1, 
               0., 
               &netin[pos+1],// will be filled with result of operation 
               1 );
  
  for (int i=1; i <= numUnits; i++) { // then: sum up
    sum += exp(netin[pos+i]);
  }
  for (int i=1; i <= numUnits; i++) { // finally: calculate outputs
    out[pos+i] = exp(netin[pos+i]) / sum;
  }
}

void MultimodalCrossEntropyOutputLayer::backwardPass(FTYPE *dedout, int copy)
{
  if (getLayerType() != OUTPUT_LAYER) {
    cerr << "Multimodal Cross Entropy Layers can only be used as output layer." << endl;
    exit(1); ///< \todo : this better should throw an exception!
  }
  
  int pos = copy*(numUnits+1);
  int posWeightMatrices = copy*(previousDim+1) * numUnits;
  
  for (int i=1; i <= numUnits; i++) {
    dEdnet[pos+i] = dEdo[pos+i];  // ATTENTION: expects (  o - t  )  in dEdo.    
    dEdo[pos+i] = (FTYPE) 0;
  }
  
  /* code that would be necessary, if this could also be used as input layer
  if (getLayerType() == INPUT_LAYER) {
    memcpy(dedout, &(dEdnet[pos+1]), sizeof (FTYPE) * numUnits);
    return; // ready. Otherwise calc derivs for weights and output of previous layer.
  } */
  
  // given dEdnet, now calculate partial derivatives for the individual weights.
  // uses a blas matrix-matrix operation to achieve this (output will be a matrix).
  cblas_dgemm( CblasRowMajor,            // Row-Major encoding of matrix
               CblasNoTrans,             // no matrix needs
               CblasNoTrans,             // to be transposed 
               numUnits, 
               previousDim+1, 
               1, 
               1., 
               &dEdnet[pos+1],           // input matrix with dEdnet
               1, 
               &net->layers[layerId-1]->out[copy*(previousDim+1)],// input with activations (actually a vector, but used here as a matrix)
               previousDim+1, 
               1.,                       // in order to sum up over the pattern!
               &dEdw[posWeightMatrices], // sum up in copy-th copy of dEdw
               previousDim+1) ; 
  
  // now sum up the partial derivatives comming from different outgoing connections
  // for each of the previous layer's neurons. This uses a BLAS 
  // matrix-vector operation.
  cblas_dgemv(CblasRowMajor, CblasTrans, numUnits, previousDim, 1., weights+1, previousDim+1, &dEdnet[pos+1], 1, 0., dedout, 1);  // skip bias
  
}



#ifdef __APPLE__
#pragma mark -
#pragma mark Individually Connected Layer
#endif


LayerArguments* IndividuallyConnectedLayer::getArguments() const
{
  return new IndividuallyConnectedLayerArguments( numCols, numRows );
}


IndividuallyConnectedLayer::IndividuallyConnectedLayer() 
: BasicLayerType(0, 0, 0, 0, 0, 0)
{
  identifer = "IndividuallyConnectedLayer";
}

IndividuallyConnectedLayer::IndividuallyConnectedLayer(Net* net, int layerId, const LayerArguments* args)
: BasicLayerType(net, layerId, args)
{
  identifer = "IndividuallyConnectedLayer";
  if (layerId > 0 &&  !net->layers[layerId-1]) {
    cerr << "Construct the previous layer and add it to the net before constructing a new IndividuallyConnectedLayer." << endl;
    exit(1); ///< \todo : better throw an exception here!
  }
}


IndividuallyConnectedLayer::~IndividuallyConnectedLayer() 
{
  if (updateFunction) {
    delete updateFunction;
  }
}

void IndividuallyConnectedLayer::forwardPass(FTYPE *input, int copy)
{
  int pos  = copy*(numUnits+1);
  for (int i=1; i <= numUnits; i++) {
    netin[pos+i] = (FTYPE) 0;
  }
  
  for (unsigned int i=0; i < connections.size(); i++) {
    netin[pos+connections[i].to] += weights[connections[i].index] * input[connections[i].from];
  }
  
  for (int i=1; i <= numUnits; i++) {
    out[pos+i] = act_f(netin[pos+i]);
  }
}

void IndividuallyConnectedLayer::backwardPass(FTYPE *dedout, int copy)
{ 
  // intialize of positions of the relevant copy for the activations in this layer, for the activations of the previous layer and for the dEdw of the kernels
  int pos = copy*(numUnits+1);

  
  for (int i=1; i <= numUnits; i++) {
    dEdnet[pos+i] = dEdo[pos+i] * deriv_f(out[pos+i], netin[pos+i]);
    dEdo[pos+i] = (FTYPE) 0;
  }
  
  if (getLayerType() == INPUT_LAYER) {
    memcpy(dedout, &(dEdnet[pos+1]), sizeof (FTYPE) * numUnits);
    return; // ready. Otherwise calc derivs for weights and output of previous layer.
  }
  
  int posPrev = copy*(net->layers[layerId-1]->numUnits+1);
  int posWeights = copy*(weights.size());
  
  for (unsigned int i=0; i < connections.size(); i++) {
    dEdw[posWeights+connections[i].index] += dEdnet[pos+connections[i].to] * net->layers[layerId-1]->out[posPrev+connections[i].from]; // ok, die Ausgabe von out[copy + 0] muesste 1 sein
      // Achtung: dedout wird nicht inklusive Bias-neuron übergeben!!!
    dedout[connections[i].from-1] += dEdnet[pos+connections[i].to] * weights[connections[i].index]; 
  }
}

void IndividuallyConnectedLayer::updateWeights(int numThreads)
{
  if (!updateFunction || !weights.size() || !variables.size()) {
    cerr << "Layer " << layerId << " not correctly initialized." << endl;
    exit(1);
  }
  
  if (numThreads > 1) {
    for (int i=1; i <= numThreads; i++) {
      cblas_daxpy(weights.size(), 1., &dEdw[weights.size()*i], 1, &dEdw[0], 1);  // Gewichständerungen zusammensummieren: N*N kernel-weights + 1 Biasgewicht
      cblas_dscal(weights.size(), 0., &dEdw[weights.size()*i], 1);           // und auf null setzen
    }
  }
  
  int numVariables = updateFunction->getNumVariables();
  
  for (unsigned int pos=0; pos < weights.size(); pos++) {  // nun alle Gewichte aller numKernels Kernel updaten
    (*updateFunction) (&weights[pos], &delta[pos], &dEdw[pos], &variables[pos*numVariables]);
  }  
}




void IndividuallyConnectedLayer::connectLayer(const BasicLayerType* previousLayer)
{  
  assert(weights.size() > 0);  // at least one weight is necessary
  
  delta.resize(weights.size(), 0.);
  dEdw.resize(weights.size() * (numCopies+1), 0.);  // n-copies, used by the n-threads to accumulate deriv. for patterns
  
  if (updateFunction) {
    setUpdateFunction(updateFunction);  // re-set the updateFunction in order to create and initialize the variable-vector
  }  
}

void IndividuallyConnectedLayer::setUpdateFunction(const UpdateFunction* updateFunction)
{
  const UpdateFunction* oldf = this->updateFunction;
  this->updateFunction = updateFunction->clone();
  
  if (oldf) {
    delete oldf;
  }

  int numVariables = this->updateFunction->getNumVariables();
  variables.resize(weights.size() * numVariables, 0.);
  for (unsigned int i=0; i < weights.size(); i++) {
    this->updateFunction->initVariables(&variables[i*numVariables]);
  }
}



void IndividuallyConnectedLayer::initWeights(int mode, FTYPE range)
{
  if(mode == 0){
    for(unsigned int i=0; i < weights.size(); i++) {
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

void IndividuallyConnectedLayer::copyWeights(const BasicLayerType* layer)
{
  const IndividuallyConnectedLayer* flayer;
  try {
    flayer = dynamic_cast<const IndividuallyConnectedLayer*> (layer);
  } catch (exception& e) {
    cerr << "Tried to copy weights from a layer of a different type." << endl;
    exit(1);
  }
  if (flayer->numUnits != numUnits) {
    cerr << "The target layer has not as many neurons as the source layer." << endl;
    exit(1);
  }
  weights = flayer->weights;
  connections = flayer->connections;
  numWeights = flayer->numWeights;
  
  IndividuallyConnectedLayer::connectLayer(net->layers[layerId-1]);
}

// input and output

void IndividuallyConnectedLayer::writeToStream(std::ostream& out) const
{
  BasicLayerType::writeToStream(out);
  
  if (getLayerType() == INPUT_LAYER) return; // no weights in input layer.
  
  out << weights.size() << " " << connections.size() << endl;
  for (unsigned int i=0; i < weights.size(); i++) {
    out << weights[i] << " ";
  }
  out << endl << endl;
  for (unsigned int i=0; i < connections.size(); i++) {
    out << connections[i].from << " " << connections[i].to << " " << connections[i].index << "   ";
  }
  out << endl;
  
}

void IndividuallyConnectedLayer::readFromStream(std::istream& in)
{
  BasicLayerType::readFromStream(in);
  
  if (layerId == 0) {
    cerr << "The layerId has to be set before reading this layer from disk. Furthermore, Convolution layers cannot be used as input layers." << endl;
    exit(1);
  }
  
  if (getLayerType() == INPUT_LAYER) return;
  
  int wsize, csize;
  in >> wsize >> csize;
  
  weights.resize(wsize);
  connections.resize(csize);
  
  for (int i=0; i < wsize; i++) {
    in >> weights[i];
  }
  for (int i=0; i < csize; i++) {
    in >> connections[i].from >> connections[i].to >> connections[i].index;
  }
  connectLayer(net->layers[layerId-1]);
}

void IndividuallyConnectedLayer::addConnection(int from, int to, int index)
{
  if (delta.size()) {
    cerr << "ERROR: add all connections BEFORE calling connect_layer." << endl;
    exit(1);
  }
  connections.push_back(Connection(from, to, index));
  if ((unsigned int)index >= weights.size()) {
    weights.resize(index+1, 0.);
  }
}


#ifdef __APPLE__
#pragma mark -
#pragma mark Convolution Layer
#endif


ConvolutionLayer::ConvolutionLayer() 
: IndividuallyConnectedLayer()
{}

ConvolutionLayer::ConvolutionLayer(Net* net, int layerId, const LayerArguments* args)
: IndividuallyConnectedLayer(net, layerId, args)
{
  const ConvolutionLayerArguments* cargs = dynamic_cast<const ConvolutionLayerArguments*> (args);
  if (layerId == 0) {
    cerr << "Don't use a ConvolutionLayer as input layer of the net." << endl;
    exit(1);
  }
  if (!net->layers[layerId-1]) {
    cerr << "Construct the previous layer and add it to the net before constructing a new convolution layer." << endl;
    exit(1);
  }
  
  kernelSize = cargs->kernelSize;
  numKernels = cargs->numKernels;
  stepsize = cargs->stepsize;
  shareWeights = cargs->shareWeights;
  overBoundary = cargs->overBoundary;
  sum = cargs->sum;
  shareBias = cargs->shareBias;
  interleave = cargs->interleave;
  
  if (cargs->kernelSize > net->layers[layerId-1]->numRows || cargs->kernelSize > net->layers[layerId-1]->numCols) {
    cerr << "ERROR: Kernel at least in one dimension is bigger than the previous layer." << endl;
    exit(1);
  }
  if (stepsize > cargs->kernelSize) {
    cerr << "WARNING: stepsize should never be bigger than the size of the kernel because otherwise there will be loose ends (neurons that are connected to nowhere)." << endl;
  }
  
  if (numUnits == 0) { // have to calculate number of units from the previous layer's size
    if (! overBoundary) {
      numCols = (net->layers[layerId-1]->numCols - 2*(kernelSize/2)-1) / stepsize + ((net->layers[layerId-1]->numCols - 2*(kernelSize/2)-1) % stepsize > 0) + 1 ;
      numCols *= (sum ? 1 : numKernels);
      if (interleave && numKernels > 1) {
        numCols = numCols / numKernels + (numCols %  numKernels > 0 ? 1 : 0) ; 
      }
      numRows = (net->layers[layerId-1]->numRows - 2*(kernelSize/2)-1) / stepsize + ((net->layers[layerId-1]->numRows - 2*(kernelSize/2)-1) % stepsize > 0) + 1 ;
      numUnits = numCols * numRows;
    }
    else {
      numCols = net->layers[layerId-1]->numCols / stepsize;
      numCols *= (sum ? 1 : numKernels);
      if (interleave && numKernels > 1) {
        numCols = numCols / numKernels + (numCols %  numKernels > 0 ? 1 : 0) ; 
      }
      numRows = net->layers[layerId-1]->numRows / stepsize;
      numUnits = numCols * numRows;
    }
    initLayer();
  }
}

void ConvolutionLayer::connectKernels(int x, int y, int to, int k)
{
 // cerr << "Connect to " << to << " from " << x << "/" << y << " for kernel " << k << endl;
  const BasicLayerType* previousLayer = net->layers[layerId-1];
  assert(to > 0);
  
  if (!sum || k==0) { // wenn summiert wird, nur beim ersten Kernel die Verbindung anlegen (sonst hätte das to Neuron mehrere Biasgewichte)
    addConnection(0, to, (shareBias ? (sum ? 0 : k) : to-1)); // nicht geteilt -> jedes neuon eigenes gewicht; geteilt && summiert -> ein einziges biasgewicht für alle; geteilt && nicht summiert -> jeder kernel eigenes gewicht
  }
  
  
  for (int ky=0; ky < kernelSize; ky++) {
    for (int kx=0; kx < kernelSize; kx++) {
        
      if (x+kx-kernelSize/2 < 0 ||  y+ky-kernelSize/2 < 0 || x+kx-kernelSize/2 >= previousLayer->numCols || y+ky-kernelSize/2 >= previousLayer->numRows) continue; // out of range
        
      int index = (shareBias ? (sum ? 1 : numKernels) : numUnits);
      if (shareWeights) {
        index += kernelSize*kernelSize*k + kx + ky*kernelSize;  // bias weights + weights of preceeding kernels + position in kernel
      }
      else {
        index += kernelSize*kernelSize * (sum ? (to-1)*numKernels : to-1) + (sum ? kernelSize*kernelSize*k : 0) + kx + ky*kernelSize;
      }
      int from =  previousLayer->calcIndexFromXY(x+kx-kernelSize/2, y+ky-kernelSize/2);
      assert(index < (int) weights.size());
      addConnection(from, to, index);
    }
  }
}


void ConvolutionLayer::connectLayer(const BasicLayerType* previousLayer)
{    
  cerr << "Creating convolutional kernels." << endl;
  
  // weight ordering: all_bias_weights_of_all_neurons, all_weights_of_kernel_0, all_weights_of_kernel_1, ...
  
  assert(net && layerId>0 && net->layers[layerId-1] && previousLayer);
  if (connections.size()) {
    cerr << "WARNING: Overwriting existing connections in order to establish convolution layer." << endl;
  }
  connections.clear();
  weights.clear();
  
  LayerTopology prevLayerTopo = LayerTopology(previousLayer->numCols, previousLayer->numRows);
  
  bool extrax = (prevLayerTopo.numCols - 2*(kernelSize/2)-1) % stepsize > 0;  // noch einen kernel extra hinten anfuegen, um alle pixel zu erwischen.
  bool extray = (prevLayerTopo.numRows - 2*(kernelSize/2)-1) % stepsize > 0;  // noch einen kernel extra hinten anfuegen, um alle pixel zu erwischen.
  
  int numx = (prevLayerTopo.numCols - 2*(kernelSize/2)-1) / stepsize + extrax + 1 ;
  int numy = (prevLayerTopo.numRows - 2*(kernelSize/2)-1) / stepsize + extray + 1 ;
  
  if (overBoundary) {
    numx = prevLayerTopo.numCols / stepsize;
    numy = prevLayerTopo.numRows / stepsize;
  }
  
  if (!sum) {
    numx *= numKernels;
  }
  
  if (interleave && numKernels > 1) {
    numx = numx / numKernels + (numx %  numKernels > 0 ? 1 : 0) ; 
  }
    
  assert (numx == numCols && numy == numRows);
  numWeights = 
    (shareBias ? (sum ? 1 : numKernels) :  numUnits) +
    (shareWeights ? kernelSize * kernelSize * numKernels :
     (sum ? kernelSize * kernelSize * numKernels * numUnits : kernelSize * kernelSize * numUnits));
       
  weights.resize(numWeights, 0.);

  /*
  // add bias connections
  for (int to=1; to <= numUnits; to++) {
    addConnection(0, to, (shareBias ? (sum ? 0 : ) : to-1));
  } */
  
  // connect kernels
  int to = 1;
  int x=0, kit=0;
  int y= overBoundary ? 0 :  kernelSize / 2;
  do {
  //  assert (to % numCols == 1);
    for (x=(overBoundary ? 0 : kernelSize / 2), kit=0 ; x < prevLayerTopo.numCols-(overBoundary ? 0 : kernelSize/2); x+=stepsize, kit = (kit+1) % numKernels) {
      if (! interleave) {
        for (int k=0; k < numKernels; k++) {
          connectKernels(x,y, to, k);
          if (!sum && k+1 < numKernels) to++;
        }
        to++;
      }
      else { // interleave  -> connect only one kernel
        if (!sum) {
          connectKernels(x,y, to, kit);
          to++;
        }
        else { // summing up
          connectKernels(x,y, to, kit);
          if (kit+1 == numKernels) to++; // only advance to the next neuron, when all kernels have been summed up.
        }
      }
    }
    if (!overBoundary && extrax) {
      if (! interleave) {
        for (int k=0; k < numKernels; k++) {
          connectKernels(prevLayerTopo.numCols-kernelSize/2-1,y, to, k);
          if (!sum && k+1 < numKernels) to++;
        }
        to++;
      }
      else { // interleave  -> connect only one kernel
        if (!sum) {
          connectKernels(prevLayerTopo.numCols-kernelSize/2-1,y, to, kit);
          to++;
        }
        else { // summing up
          connectKernels(prevLayerTopo.numCols-kernelSize/2-1,y, to, kit);
          if (kit+1 == numKernels) to++; // only advance to the next neuron, when all kernels have been summed up.
        }
      }
    }
    
    y += stepsize;
    if (y >= prevLayerTopo.numRows-(overBoundary ? 0 : kernelSize/2) && !overBoundary && extray) {  // noch einen kernel extra hinten anfuegen, um alle pixel zu erwischen? 
      y = prevLayerTopo.numRows-kernelSize/2-1;
      extray = false;
    }
  } while (y < prevLayerTopo.numRows-(overBoundary ? 0 : kernelSize/2));
  

  cerr << connections.size() << " == " << numUnits << " + " << numUnits << " * " << kernelSize << "^2 * " << (sum ? numKernels : 1) << " = " 
                             << numUnits + (numUnits * kernelSize * kernelSize * (sum ? numKernels : 1)) << endl;
  assert (overBoundary || (int)connections.size() == numUnits + (numUnits * kernelSize * kernelSize * (sum ? numKernels : 1)));
  assert((int)weights.size() == numWeights);
  
  IndividuallyConnectedLayer::connectLayer(previousLayer);  // call base-class' connect_layer to fix other matrices and updateFunction.
  
  
  // now run some tests...
  
  for (int unit = 1; unit <= numUnits; unit++) {
    int x, y;
    calcXYFromIndex(unit, &x, &y);
    // cerr << x << " / " << y << endl;
    
    int wx=kernelSize, wy=kernelSize;
    if (x-kernelSize/2 < 0) {
      wx -= kernelSize/2-x;
    }
    if (y-kernelSize/2 < 0) {
      wy -= kernelSize/2-y;
    }    
    if (x+kernelSize/2 > numCols-1) {
      wx -= x+kernelSize/2 - (numCols-1);
    }
    if (y+kernelSize/2 > numRows-1) {
      wy -= y+kernelSize/2 - (numRows-1);
    }
    int numConnections = (sum ? numKernels : 1) * wx * wy + 1;
    int cc = 0;
    int bc = 0;
    for (unsigned int i=0; i < connections.size(); i++) {
      if (connections[i].to == unit) {
        cc++;
        if (connections[i].from == 0) {
          bc++;
        }
      }
    }
    assert (bc == 1);
    if (overBoundary == 1 && numKernels == 1 && stepsize == 1 &&  cc != numConnections) {
      cerr << unit << " " << x << "," << y << " -> widthx: " << wx << " widthy: " << wy << "   should: " << numConnections << " but has: " << cc << endl;
      for (unsigned int i=0; i < connections.size(); i++) {
        if (connections[i].to == unit) {
          cerr << "-> from " << connections[i].from << " index " << connections[i].index << endl;
          cc++;
          if (connections[i].from == 0) {
            bc++;
          }
        }
      }
      cerr << endl;
    }
    
  }
}



#ifdef __APPLE__
#pragma mark -
#pragma mark Inverted Layer
#endif

InvertedLayer::InvertedLayer() 
: IndividuallyConnectedLayer()
{}

InvertedLayer::InvertedLayer(Net* net, int layerId, const LayerArguments* args)
: IndividuallyConnectedLayer(net, layerId, args)
{
  if (layerId == 0) {
    cerr << "Don't use a Inverted-layer as input layer of the net." << endl;
    exit(1);
  }
  if (!net->layers[layerId-1]) {
    cerr << "Construct the previous layer and add it to the net before constructing a new convolution layer." << endl;
    exit(1);
  }
}

void InvertedLayer::createInvertedWeights(const IndividuallyConnectedLayer* ilayer)
{      
  assert(net && layerId>0 && net->layers[layerId-1] && ilayer && ilayer->net && ilayer->net->layers[layerId-1]);
  const BasicLayerType* previousLayer = net->layers[layerId-1];
  assert(previousLayer->numUnits == ilayer->numUnits);
  assert(ilayer->net->layers[ilayer->layerId-1]->numUnits == numUnits);
  
  if (connections.size()) {
    cerr << "WARNING: Overwriting existing connections in order to establish convolution layer." << endl;
  }
  connections.clear();
  weights.clear();
      
  
  // add bias connections
  for (int to=1; to <= numUnits; to++) {
    addConnection(0, to, to-1);    //  0); <-  shared  |  separate -> to-1);
  }
  
  numWeights = connections.size();
  
  // bias weights are unknown, and may be different. Therefore recreate all index values in this layer (ordering may change arbitrarily)
  map<int, int> indexMap;   // key: old index, value: new index
  for (unsigned int i=0; i < ilayer->connections.size(); i++) {
    if (ilayer->connections[i].from != 0) { // not a bias weight
      if (indexMap.find(ilayer->connections[i].index) == indexMap.end()) {  // not already in the mapping
        indexMap[ilayer->connections[i].index] = numWeights++;  // add to the mapping, use present weight index, increase index by one.
      }
    }
  }
  
  // now add the connections using the mapping for translating indices. 
  for (unsigned int i=0; i < ilayer->connections.size(); i++) { 
    if (ilayer->connections[i].from != 0) { // ignore bias-weights
      addConnection(ilayer->connections[i].to, ilayer->connections[i].from, indexMap[ilayer->connections[i].index]);
    }
  }
    
  assert((int)weights.size() == numWeights);
  
  IndividuallyConnectedLayer::connectLayer(previousLayer);  // call base-class' connect_layer to fix other matrices and updateFunction.
}




REGISTER_LAYERTYPE(MultimodalCrossEntropyOutputLayer, "Same as fully connected layer but with special backprop for softmax activation and cross entropy error.")
REGISTER_LAYERTYPE(IndividuallyConnectedLayer, "N++-like individually linked layer.")
REGISTER_LAYERTYPE(InvertedLayer, "N++-like individually linked layer inverting the weight-structure of another IndividuallyConnectedLayer for the use in an auto-encoder.")
REGISTER_LAYERTYPE(ConvolutionLayer, "Special layer realizing several convolutional kernels in a n++-style.")

