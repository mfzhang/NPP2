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
 *  LayerTypes.cpp
 *  Created by Sascha Lange on 30.04.09.
 */

#include "BasicLayerTypes.h"
#include "npp2.h"
#include <cstdlib>
#include <iostream>
#include <functions.h>
#include "PatternSet.h"
#include "Registry.h"
#include <cassert>


using namespace NPP2;
using namespace std;


BasicLayerType::BasicLayerArguments::BasicLayerArguments( int numCols, int numRows) 
: LayerArguments("BasicLayerType"), numCols(numCols), numRows(numRows) 
{}

void BasicLayerType::setActivationFunction(int actId)
{
  switch (actId) {  
    case NPP_LINEAR: {
      this->actId = NPP_LINEAR;
      this->act_f = linear;
      this->deriv_f = linear_deriv;
    } break;
    case NPP_LOGISTIC:
    default: {
      this->actId = NPP_LOGISTIC; // everything != NPP_LINEAR is mapped to NPP_LOGISTIC
      this->act_f = logistic;
      this->deriv_f = logistic_deriv;
    } break;
  }
}



BasicLayerType::BasicLayerType(Net* net, int layerId, const LayerArguments* args)
: identifer("BasicLayerType"), net(net), layerId(layerId), firstUnitId(0), numWeights(0), updateFunction(0)
{
  const BasicLayerType::BasicLayerArguments* bargs = dynamic_cast<const BasicLayerType::BasicLayerArguments*> (args);

  numUnits  = bargs->numRows * bargs->numCols;
  numRows   = bargs->numRows;
  numCols   = bargs->numCols;
  numCopies = net->numCopies;
  
  actId = NPP_LOGISTIC;   // default is logistic activation
  
  if (numUnits) {
    initLayer();          // this will set act_f and deriv_f
  }
}


BasicLayerType::BasicLayerType(Net* net, int layerId, int firstUnitId, int unitsPerRow, int numRows, int numCopies)
: identifer("BasicLayerType"), net(net), layerId(layerId), firstUnitId(firstUnitId), numUnits(unitsPerRow * numRows), numRows(numRows), 
numCols(unitsPerRow), numCopies(numCopies), numWeights(0), updateFunction(0)
{
  actId = NPP_LOGISTIC;
  
  if (numUnits > 0) {
    initLayer();
  }
}


void BasicLayerType::initLayer()
{
  if (numUnits > 0) {
  
    // create and zero-initialize the needed vectors to hold net-input, 
    // output, and partial derivatives for each neuron. there is one entry for
    // each of the neurons (including the bias neuron 0) in each of the copies. 
    dEdo   = new FTYPE[(numUnits+1)*(numCopies+1)];
    dEdnet = new FTYPE[(numUnits+1)*(numCopies+1)]; 
    out    = new FTYPE[(numUnits+1)*(numCopies+1)]; 
    netin  = new FTYPE[(numUnits+1)*(numCopies+1)];
  
    memset(dEdo,   0, sizeof(FTYPE) * (numUnits+1)*(numCopies+1));
    memset(dEdnet, 0, sizeof(FTYPE) * (numUnits+1)*(numCopies+1));
    memset(out,    0, sizeof(FTYPE) * (numUnits+1)*(numCopies+1));
    memset(netin,  0, sizeof(FTYPE) * (numUnits+1)*(numCopies+1));
  
    for (int i=0; i < numCopies+1; i++) {  // set bias weight to 1
      out[i * (numUnits+1)] = (FTYPE) 1.;
    }
    
    // set activation function and its derivative. this could also 
    // be replaced by an abstract factory (registry) in the future.
    if (actId == NPP_LOGISTIC) { 
      act_f = logistic;
      deriv_f = logistic_deriv;
    }
    else if (actId == NPP_LINEAR) {
      act_f = linear;
      deriv_f = linear_deriv;
    }
    else {
      cerr << "Unknown activation function: " << actId << endl;
      exit(1);
    }
  }
}

BasicLayerType::~BasicLayerType()
{
  if (numUnits > 0) {
    delete [] dEdo;
    delete [] dEdnet;
    delete [] out;
    delete [] netin;
  }
}

int BasicLayerType::getLayerType() const 
{ 
  return layerId == 0 ? INPUT_LAYER : net->topoData.layerCount-1 == layerId ? OUTPUT_LAYER : HIDDEN_LAYER; 
}
