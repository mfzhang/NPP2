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


/*
 *  n++2.cpp
 *  n++2
 *
 *  Created by Sascha Lange on 28.04.09.
 *  Copyright 2009-2011 Sascha Lange. All rights reserved.
 *
 */

#include "npp2.h"
#include <cstdlib>
#include <iostream>
#include <functions.h>
#include "PatternSet.h"

#if  (defined(__APPLE_CPP__) || defined(__APPLE_CC__) || defined(__MACOS_CLASSIC__))
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <fstream>
#include <vector>
#include "Registry.h"
#include "BasicLayerTypes.h"
#include "FullyConnectedLayer.h"
#include <cassert>

using namespace NPP2;
using namespace std;

enum ReturnType { NPP_ERROR=0, NPP_OK=1 };           ///< return type of methods

#ifdef __APPLE__
#pragma mark -
#pragma mark Propagation and weight adaptation 
#endif

// forward propagation function for calculating activations. the parameter 'copy' specifies the copy of the network to work on.
void Net::forwardPass(const FTYPE *inVec, FTYPE *outVec, int copy)
{
  assert (topoData.inCount == layers[0]->numUnits);
  
  memcpy(&(layers[0]->out[copy*(layers[0]->numUnits+1)+1]), // copy specified network input to output of first layer
         inVec, 
         sizeof(FTYPE) * topoData.inCount);
  
  for (int i=1; i < topoData.layerCount; i++) {             // layer-wise propagation
    layers[i]->forwardPass(&(layers[i-1]->out[copy*(layers[i-1]->numUnits+1)]), copy);
  }
  
  memcpy(outVec,                                            // copy the calculated ouptut from the last layer to the right part of the output vector
         &(layers[topoData.layerCount-1]->out[copy*(topoData.outCount+1)+1]),
         sizeof(FTYPE) * topoData.outCount);
}

// backward propagation function for calculating partial derivatives. the parameter 'copy' specifies the copy of the network to work on.
void Net::backwardPass(const FTYPE *dedout, FTYPE *dedin, int copy)
{
  assert ( topoData.outCount == layers[topoData.layerCount-1]->numUnits);
  
  memcpy(&(layers[topoData.layerCount-1]->dEdo[copy*(topoData.outCount+1)+1]),  // copy specified part. drivative de/dout to the output layer of the network
         dedout, 
         sizeof(FTYPE) * topoData.outCount);
  
  for (int i=topoData.layerCount-1; i > 0; i--) {          // back propagate error through the layers
    layers[i]->backwardPass(&(layers[i-1]->dEdo[copy*(layers[i-1]->numUnits+1)+1]), copy);
  }
  
  layers[0]->backwardPass(dedin, copy);                     // in input layer write derivatives into given dedin argument. there is no copy of dedin at layer zero, like there is with in_vec for the input
}

void Net::updateWeights(int numThreads) 
{
  for (int i=1; i < topoData.layerCount; i++) {// loop through all layers and
    layers[i]->updateWeights(numThreads);      // tell them to update their weights
  }
}

#ifdef __APPLE__
#pragma mark -
#pragma mark Network creation and initialization
#endif 

// adds a layer to the present end of the network
// needs to adapt topology info and size of output vector appropriately
void Net::addLayer(BasicLayerType* layer)
{
  BasicLayerType* previousLayer = topoData.layerCount > 0 ? layers[topoData.layerCount-1] : 0; // last layer of present net, or '0', iff net is empty
  int unitid = previousLayer ? previousLayer->firstUnitId+previousLayer->numUnits-1+1 : 1;  // determine the next unit id to use (starts with 1, as zero is used for the bias weight)
  
  layers.resize(topoData.layerCount+1, 0);       // make space for additional layer
  layers[topoData.layerCount] = layer;           // add new layer to the end

  assert(layer->layerId == topoData.layerCount);
  assert(layer->net == this);
  assert(layer->numCopies == numCopies);
  
  layer->firstUnitId = unitid;                   // assign unit id
  
  if (updateFunction && topoData.layerCount > 0) { // set update function of net
    layer->setUpdateFunction(updateFunction);
  }
  
  // if this is the input layer, we have to set a view special
  // parameters that make sure correct calculation for the case,
  // where nets have been 'stacked' and this input layer is 'identical'
  // to the output layer of a previous net. using linear activation asures
  // that it's possible to simply copy derivatives calculated at this input
  // layer can be passed to the other net's output layer (inserted as dEdo).
  if (topoData.layerCount == 0) { 
    layer->act_f = linear;
    layer->deriv_f = linear_deriv;
    layer->actId = NPP_LINEAR;
    
    topoData.inCount = layer->numUnits; // also correct size of input layer
    
    if (inVec) delete [] inVec;         // and resize input vector appropriately
    inVec = new FTYPE [topoData.inCount * (numCopies+1)];
    if (numCopies > 0) {
      if (workerData) delete [] workerData;
      workerData = new WorkerData[numCopies];
    }
  }
  
  // now correct size of output layer and output vector
  topoData.outCount = layer->numUnits;
  if (outVec) delete [] outVec;
  outVec = new FTYPE [topoData.outCount * (numCopies+1)];
  
  topoData.layerCount+=1;
}


// uses the previous method to add a newly constructed layer
// with the given args to the end of the net.
void Net::addLayer(LayerArguments* args, bool cleanUpArgs)
{
  LayerFactory::Arguments largs (this, topoData.layerCount, args->identifer, args);
  // now construct a layer with the help of the layer factory. this 
  // construction is 'abstract', in the sense that we don't know the exact
  // type of the resulting layer.
  BasicLayerType* layer = LayerFactory::getTheLayerFactory()->create(&largs);
  if (!layer) {  // layer construction failed for unknown reasons.
    char buf[10000];
    LayerFactory::getTheLayerFactory()->listEntries(buf, 10000);
    cerr << "ERROR: Could not find a matching class for requested layer of type " << args->identifer << "." << endl;
    cerr << "The available layer types are " << endl;
    cerr << buf << endl;
    exit(1);
  }
  
  addLayer(layer);
  
  if (cleanUpArgs) {
    delete args;
  }
}

// creates a net with FullyConnectedLayers using the given layer sizes.
void Net::createLayers(int numLayers, int *layerUnits, int numCopies) throw (NPPException)
{
  if(inVec){
    cerr << "Can't create layers - network already defined" << endl;
    throw NPPException("Error creating layers: network already defined."); 
  }
  this->numCopies = numCopies;
  
  topoData.layerCount = numLayers;
  topoData.inCount = layerUnits[0];
  topoData.outCount = layerUnits[numLayers-1];
  
  int unitid = 1;
  
  layers.resize(numLayers, 0);
  for(int i=0; i<topoData.layerCount;i++){
    layers[i] = new FullyConnectedLayer(this, i, unitid, layerUnits[i], 1, numCopies);
    
    if (i == 0) { // input layer linear activation
      layers[i]->act_f = linear;
      layers[i]->deriv_f = linear_deriv;
      layers[i]->actId = NPP_LINEAR;
    }
    if (updateFunction && i > 0) {  // setUpdateFunc may be called before or after createLayers!
      layers[i]->setUpdateFunction(updateFunction);
    }
  }
  inVec = new FTYPE [topoData.inCount * (numCopies+1)];
  outVec = new FTYPE [topoData.outCount * (numCopies+1)];
  
  if (numCopies > 0) workerData = new WorkerData[numCopies];
  
}



// creates a complete net according to the given list of layer-arguments.
void Net::createLayers(const std::vector<LayerArguments*>& netSpecification, int numCopies, bool cleanUpArgs) throw (NPPException)
{
  assert(netSpecification.size() >= 1);
  if(inVec){
    cerr << "Can't create layers - network already defined" << endl;
    throw NPPException("Error creating layers: network already defined."); 
  }
  this->numCopies = numCopies;
  
  topoData.layerCount = netSpecification.size();
  layers.resize(topoData.layerCount, 0);
  
  int unitid = 1;
  for (unsigned int i=0; i < netSpecification.size(); i++) {
    LayerFactory::Arguments largs (this, i, netSpecification[i]->identifer, netSpecification[i]);
    layers[i] = LayerFactory::getTheLayerFactory()->create(&largs);
    if (!layers[i]) {
      char buf[10000];
      LayerFactory::getTheLayerFactory()->listEntries(buf, 10000);
      cerr << "ERROR: Could not find a matching class for requested layer of type " << netSpecification[i]->identifer << "." << endl;
      cerr << "The available layer types are " << endl;
      cerr << buf << endl;
      exit(1);
    }
      
    layers[i]->firstUnitId = unitid;
    unitid += layers[i]->numUnits;
    
    if (updateFunction && i > 0) { // setUpdateFunc may be called before or after createLayers!
      layers[i]->setUpdateFunction(updateFunction);
    }
  }
  
  // input layer always linear activation
  layers[0]->act_f = linear;
  layers[0]->deriv_f = linear_deriv;
  layers[0]->actId = NPP_LINEAR;
  
  topoData.inCount = layers[0]->numUnits;
  topoData.outCount = layers[topoData.layerCount-1]->numUnits;
  
  inVec = new FTYPE [topoData.inCount * (numCopies+1)];
  outVec = new FTYPE [topoData.outCount * (numCopies+1)];
  
  if (numCopies > 0) workerData = new WorkerData[numCopies];
  
  if (cleanUpArgs) {
    for (unsigned int i=0; i < netSpecification.size(); i++) {
      delete netSpecification[i];
    }
  }
}




void Net::setLayerActivationFunction(int layerNo, int actId)
{
  if (layerNo < 0 || layerNo >= topoData.layerCount) {
    cerr << "Should set the activation function of layer " << layerNo
         << ". This net only has " << topoData.layerCount << " layers." << endl;
    exit(1);
  }
  layers[layerNo]->setActivationFunction(actId);
}


void Net::setUpdateFunc(int typ, FTYPE *params)
{
  for (int j=0; j < MAX_PARAMS; j++) {
    updateParams[j] = params[j];
  }
  if (updateFunction) {
    delete updateFunction;
  }
  if (typ == 0)  { // RPROP
    updateFunction = new RPROP(params);
  }
  else {
    cerr << "Update function " << typ << " unknown." << endl;
    exit(1);
  }
  if (layers.size()) {
    for (int i=1; i < topoData.layerCount; i++) {
      layers[i]->setUpdateFunction(updateFunction);
    }
  }
}


void Net::connectLayers()
{
  for (int i=1; i < topoData.layerCount; i++) {
    layers[i]->connectLayer(layers[i-1]); // connects the layer to the PREVIOUS layer
  }
}

void Net::initWeights(int mode, FTYPE range) {
  for (int i=1; i < topoData.layerCount; i++) {
    layers[i]->initWeights(mode, range); // inits the weights of connections TO this layer
  }  
}

#ifdef __APPLE__
#pragma mark -
#pragma mark Object life cycle
#endif 

void Net::deleteStructure()
{
  if (inVec) {
    delete [] inVec; inVec = 0;
    delete [] outVec; outVec = 0;
    
    for (int i=0; i < topoData.layerCount; i++) {
      if (layers[i]) delete layers[i];  // maybe only partially initialized in case of an error
    }
    layers.clear();

  }
  if (updateFunction) {
    delete updateFunction;
    updateFunction = 0;
  }
}

Net::~Net() 
{
  deleteStructure();
}

Net::Net(int numCopies) : inVec(0), outVec(0), layers(0), updateFunction(0), numCopies(numCopies), workerData(0)
{
  topoData.layerCount = 0;
  topoData.inCount = topoData.outCount = 0;
}


#ifdef __APPLE__
#pragma mark -
#pragma mark Training and testing
#endif 

double Net::train(const PatternSet* pattern, const ErrorFunction* errorFunction, bool id, int numMiniBatches)
{
  return train(pattern, numCopies, id, errorFunction, numMiniBatches);
}

double Net::train(const PatternSet* pattern, int threads, bool id, const ErrorFunction* errorFunction, int numMiniBatches)
{
  
  if (threads <= 1) {  // simple version for single-threaded nets
    double tss=0.;
    int perBatch = pattern->pattern_count / numMiniBatches;

    // implements (optional) mini-batches: updates the weights several
    // times during one iteration over all patterns; once after propagating
    // a (smaller) fraction of the total pattern set.
    for (int batch = 0; batch < numMiniBatches; batch++) { 
      for (int i=perBatch * batch; i < pattern->pattern_count && (i < perBatch*(batch+1) || batch == numMiniBatches-1); i++) {  // process remainder in last batch
        forwardPass(pattern->input[i], outVec); // propagate activation through net
    
        FTYPE* target = id ? pattern->input[i] : pattern->target[i]; // the 'id' option can be used when training an auto-encoder; id -> target == input
      
        for (int d=0; d < topoData.outCount; d++) {
          tss += errorFunction->error(outVec[d], target[d]);
          outVec[d] = errorFunction->deriv(outVec[d], target[d]);    // just calculate the partial derviative for the output: out_vec := dE/do = (o-t) 
        }
        backwardPass(outVec, inVec);            // back-propagate error-derivatives 
      }
      updateWeights();                          // finally update the weights
    }
    return tss;
  }
  else { // this is a threaded version that works on multiple copies of the net
    if (threads > numCopies) {
      cerr << "Asked to start " << threads << " threads but only have " 
           << numCopies << " copies of network. Not possible." << endl; 
      return -1.;
    }

    double tss = 0.;
    for (int batch = 0; batch < numMiniBatches; batch++) {
      for (int i=0; i < threads; i++) { // prepare the data for the workers that'll work in parallel, each on a fraction of the training patterns 
        workerData[i] = WorkerData(this, errorFunction, pattern, i, threads, id, batch, numMiniBatches);
        if (i==threads-1) trainWorker(&workerData[i]); // last fraction will be done by this (main) thread
        else pthread_create(&(workerData[i].threadId), 0, Net::trainWorker, (void*) &workerData[i]); // start thread for each worker
      }
      for (int i=0; i < threads-1; i++) { // now wait for all threads to finish
        pthread_join(workerData[i].threadId, 0);
        tss += workerData[i].tss;
      }
      tss+=workerData[threads-1].tss;     // don't forget the error accumulated in this (main) thread
      updateWeights(threads);             // finally update the weights
    }
    return tss;
  }
}

// this is the train-version used by a single worker-thread
// it's practically identical to the single-thread version above
// the only differences are
// a) it works on it's own copy of the network's weights and activations (pos)
// b) it only processes a fraction of the training pattern; therefore the more complicated loop over the pattern.
void Net::trainWorker(WorkerData* arg)
{
  int pos = (arg->thread+1) * topoData.outCount;
  int perBatch = arg->pattern->pattern_count / arg->numMiniBatches;

  for (int i=perBatch * arg->batch + arg->thread; 
       i < arg->pattern->pattern_count && (i < perBatch*(arg->batch+1)+arg->thread || arg->batch == arg->numMiniBatches-1); 
       i+= arg->numThreads) {
    forwardPass(arg->pattern->input[i], &outVec[pos], arg->thread+1);
    
    FTYPE* target = arg->trainId ? arg->pattern->input[i] : arg->pattern->target[i];
    
    for (int d=0; d < topoData.outCount; d++) {
      arg->tss += arg->errorFunction->error(outVec[pos+d], target[d]); 
      outVec[pos+d] = arg->errorFunction->deriv(outVec[pos+d], target[d]); 
    }
    backwardPass(&outVec[pos], &inVec[pos], arg->thread+1);
  }
}


// static function to be called by pthread create. then send's 
// the thread back to the object's train method
void* Net::trainWorker(void* arg)
{
  WorkerData* argl = (WorkerData*) arg;
  argl->net->trainWorker(argl); 
  pthread_exit(0);
}




Error Net::test(const PatternSet* pattern, const ErrorFunction* errorFunction, bool id)
{
  return test(pattern, numCopies, id, errorFunction);
}

Error Net::test(const PatternSet* pattern, int threads, bool id, const ErrorFunction* errorFunction)
{
  if (threads <= 1) { // single threaded test version. no backprop, no updates
    Error error;
    error.regrError = 0.;
    int countwrong=0;
    for (int i=0; i < pattern->pattern_count; i++) {
      forwardPass(pattern->input[i], outVec);
      
      FTYPE* target = id ? pattern->input[i] : pattern->target[i];
      
      int outI=-1, targetI=-1; double targetMax=0., outMax=0.;
      for (int d=0; d < topoData.outCount; d++) {
        error.regrError += errorFunction->error(outVec[d], target[d]);
        if (outVec[d] >= outMax) {
          outMax = outVec[d];
          outI = d;
        }
        if (target[d] >= targetMax) {
          targetMax = target[d];
          targetI = d;
        }
      }
      if (targetI != outI) countwrong++;
    }
    error.classError = (countwrong / (double)pattern->pattern_count) * 100.;
    return error;
  }
  else {  // multi threaded version. same logic as in train, but no updates, no backprop.
    if (threads > numCopies) {
      cerr << "Asked to start " << threads << " threads but only have " 
      << numCopies << " copies of network. Not possible." << endl; 
      return Error();
    }
    
    for (int i=0; i < threads; i++) {
      workerData[i] = WorkerData(this, errorFunction, pattern, i, threads, id);
      if (i==threads-1) testWorker(&workerData[i]);
      else pthread_create(&(workerData[i].threadId), 0, Net::testWorker, (void*) &workerData[i]);
    }
    Error error;
    error.regrError = 0.;
    int countwrong = 0;
    for (int i=0; i < threads-1; i++) {
      pthread_join(workerData[i].threadId, 0);
      error.regrError += workerData[i].tss;
      countwrong += workerData[i].countwrong;
    }
    error.regrError+=workerData[threads-1].tss;
    countwrong+=workerData[threads-1].countwrong;
    error.classError = (countwrong / (double)pattern->pattern_count) * 100.;

    return error;
  }
}

void Net::testWorker(WorkerData* arg)
{
  int pos = (arg->thread+1) * topoData.outCount;
  for (int i=arg->thread; i < arg->pattern->pattern_count; i+= arg->numThreads) {
    forwardPass(arg->pattern->input[i], &outVec[pos], arg->thread+1);
    
    FTYPE* target = arg->trainId ? arg->pattern->input[i] : arg->pattern->target[i];
    
    int outI=-1, targetI=-1; double targetMax=0., outMax=0.;
    for (int d=0; d < topoData.outCount; d++) {
      arg->tss += arg->errorFunction->error(outVec[pos+d], target[d]); 
      if (outVec[d] >= outMax) {
        outMax = outVec[d];
        outI = d;
      }
      if (target[d] >= targetMax) {
        targetMax = target[d];
        targetI = d;
      }
    }
    if (targetI != outI) arg->countwrong++;
  }
}



void* Net::testWorker(void* arg)
{
  WorkerData* argl = (WorkerData*) arg;
  argl->net->testWorker(argl);
  pthread_exit(0);
}

#ifdef __APPLE__
#pragma mark -
#pragma mark Saving and loading
#endif 


// the format used for saving the nets is not the same as in the original n++.
// there is a header, and then each layer is encoded in a type-specific
// manner; that is, each layer type has its own format for placing the
// parameters and weights.
void Net::saveNet(std::ostream& out) const throw (NPPException)
{ 
  out << "# Layers: " << topoData.layerCount << endl;
  out << "# Topology (input - hidden - output)\n";
  out << "topology: ";
  for(int l= 0; l<topoData.layerCount; l++)
    out << layers[l]->numUnits << " ";
  out << endl << "set_update_f " << 0 << " "; //update_id << " ";
  for(int i= 0; i<MAX_PARAMS; i++)
    out << updateParams[i] << " ";
  out << endl << endl;
  
  out << "# Layerwise connection structure - principle format:\n";
  out << "# <layerId> <BasicData> <TypeSpecificData>\n" << endl;
  
  out << "layer_def" << endl << endl;
  
  for (int i=0; i < topoData.layerCount; i++) {
    out << " " << *layers[i] << endl << endl;
  }
}

#define MAX_LINE_LEN 4096

void Net::loadNet(std::istream& in) throw (NPPException)
{
  char line[MAX_LINE_LEN];
  char *value;
  vector<int> topo;
  int i;
  int define_new;   /* type of network description */
  int mode,no;
  FTYPE range;
  double p[MAX_PARAMS];
  
  if (layers.size()) {
    cerr << "Net defined - OVERWRITING" << endl;
    deleteStructure(); /* delete old net */
  }
  define_new = 0;   /* type of description not yet identified */
  // int t=0;
  while(in && in.getline(line, MAX_LINE_LEN-1) &&
        strncmp(line, "NetEnd", 6)!=0) {
    
    line[strlen(line)+1] = '\0';    // slange: add linebreak, 
    line[strlen(line)] = '\n';      //         to mimic fgets behavior
    
    if(*line=='\n'||*line=='#');  /* skip comments */
    
    else if (strncmp(line,"topology",7)==0){
      value=strtok(line, " \t\n");  /* skip first token (== topology)*/
      for(i=0,value=strtok( NULL," \t" );(value!=NULL); value=strtok( NULL," \t\n" ),i++) {
        topo.push_back((int)atoi(value));
      } /* finished reading topology  */
    }
    else if (strncmp(line,"set_update_f",12)==0){
      value=strtok(line, " \t\n");  /* skip first token (== set_update_f)*/
      value=strtok( NULL," \t\n" ); /* second token = type of update fun */
      mode = (int) atoi(value);
      for(i=0,value=strtok( NULL," \t" );(value!=NULL)&&(i<MAX_PARAMS);
          value=strtok( NULL," \t\n" ),i++){
        p[i] = (double)atof(value);
      } /* finished reading update params  */
      setUpdateFunc(mode,p);
    }
    else if (strncmp(line,"connect_layers",12)==0) {
      if (topo.size() == 0) {
        cerr << "ERROR: Topology not specified before call to connect layers." << endl;
        throw NPPException("Error parsing network file: topology was not specified before command to connect layers.");
      }
      createLayers(topo.size(), &topo[0]);
      connectLayers();
      define_new = 1;
    }
    else if (strncmp(line,"connect_shortcut",12)==0){
      cerr << "ERROR: Shortcut connections are not allowed in this implementation." << endl;
      throw NPPException("Error parsing network file: no shortcut connection allowed.");
    }
    else if (strncmp(line,"connect_units",12)==0){
      cerr << "ERROR: Specifying individual connections is not supported by this implementation." << endl;
      throw NPPException("Error parsing network file: specifying of individual connections is not allowed in n++2.");
    }      
    else if (strncmp(line,"init_weights",10)==0){
      if (topo.size() == 0) {
        cerr << "ERROR: Topology not specified before call to init weights." << endl;
        throw NPPException("Error parsing network file: topology was not specified before command to initialize weights.");
      }
      sscanf(line,"%*s %d %lf",&mode,&range);
      initWeights(mode,range);
    }
    else if (strncmp(line,"set_layer_act_f",12)==0) {
      if (topo.size() == 0) {
        cerr << "ERROR: Topology not specified before call to set layer activation function." << endl;
        throw NPPException("Error parsing network file: topology was not specified before setting the layer activation function.");
      }
      sscanf(line,"%*s %d %d",&no,&mode);
      setLayerActivationFunction(no,mode);
    }      
    else if (strncmp(line,"set_unit_act_f",12)==0){
      cerr << "ERROR: Specifying activation functions for individual units is not supported by this implementation." << endl;
      throw NPPException("Error parsing network file: specifying activation function of individual units is not allowed in n++2.");
    }
    else if (strncmp(line,"input_scale",10)==0){
      cerr << "ERROR: Scaling is not yet supported by this implementation." << endl;
      throw NPPException("Error parsing network file: output and input scaling is not yet implemented.");
    }      
    else if (strncmp(line,"output_scale",10)==0){
      cerr << "ERROR: Scaling is not yet supported by this implementation." << endl;
      throw NPPException("Error parsing network file: output and input scaling is not yet implemented.");
    }      
    else if (strncmp(line, "layer_def", 9)==0) { /* read  weights from file */      
      if (define_new) {
        cerr << "ERROR: Either specify weights or call connect layers." << endl;
        throw NPPException("Error parsing network file: either specify weights or call connect layers.");
      }
      if (topo.size() == 0) {
        cerr << "ERROR: Specify topology before start of layer definitions." << endl;
        throw NPPException("Error parsing network file: topology was not specified before start of weight definitions.");
      }
      topoData.inCount = topo[0];
      topoData.outCount = topo[topo.size()-1];
      topoData.layerCount = topo.size();
      
      layers.resize(topoData.layerCount);
      
      for (int i=0; i < topoData.layerCount; i++) {
        int id;
        string name;
        in >> id >> name;
                
        if (id != i) {
          cerr << "ERROR PARSING LAYERS: Id does not match position of layer." << endl;
          throw NPPException("Error parsing network file: wrong order in definition of layers.");
        }
        
        LayerFactory* factory = LayerFactory::getTheLayerFactory();
        layers[i] = factory->create(name);  // returns 0 iff layer type specified by "name" not found
        
        if (layers[i] == 0) {
          cerr << "ERROR PARSING LAYERS: Do not know specified layer type: " << name << "." << endl;
          throw NPPException("Error parsing network file: layer of unknown type (type not supported by this build).");
        }
        layers[i]->net = this;
        layers[i]->layerId = id;
        in >> *(layers[i]);
        if (updateFunction) {
          layers[i]->setUpdateFunction(updateFunction);
        }
      }
      
      numCopies = layers[0]->numCopies;
      
      inVec = new FTYPE [topoData.inCount * (numCopies+1)];
      outVec = new FTYPE [topoData.outCount * (numCopies+1)];
      
      if (numCopies > 0) workerData = new WorkerData[numCopies];
        
    } /* if units already defined */
  } /* while read line from file */
}

void Net::saveNet(const std::string& filename) const throw (NPPException)
{
  ofstream out (filename.c_str());
  if (!out) throw NPPException("Could not open network file for writing.");
  saveNet(out);
  out.close();
}

void Net::loadNet(const std::string& filename) throw (NPPException)
{
  ifstream in(filename.c_str());
  if (!in) throw NPPException("Could not open network file for reading.");
  loadNet(in);
  in.close();
}




