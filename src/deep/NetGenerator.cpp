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

/*  NetGenerator.cpp
 *  Created by Sascha Lange on 25.11.08.
 */

#include "NetGenerator.h"
#include "BasicLayerTypes.h"
#include "AdvancedLayerTypes.h"
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include "npp2.h"
#include <cstdlib>

using namespace std;
using namespace NPP2;

ConvolutionNetGenerator::ConvolutionNetGenerator(int width, int height, int numCopies, int fieldWidth, bool shareWeights, int numKernels, double reductionBase, bool shareBias, bool overlapping, int numReceptLayers) :
  width(width), height(height), numCopies(numCopies), fieldWidth(fieldWidth), shareWeights(shareWeights), numKernels(numKernels), reductionBase(reductionBase), shareBias(shareBias), overlapping(overlapping), numReceptLayers(numReceptLayers)
{
  assert(reductionBase > 1.);
  assert(width > 0 && height > 0);
  assert(fieldWidth % 2 == 1);
}

ConvolutionNetGenerator::~ConvolutionNetGenerator()
{}


Net* ConvolutionNetGenerator::generateNet(int targetDim, int secondLast) const
{
  // Ziel: In ersetm hidden layer die anzahl auf das 1,5-fache aufblaehen, 
  //       dann in jeder Schicht anzahl der neuronen halbieren, bis hin zu
  //       der letzten schicht, die targetDim neuronen hat.
  // die anfangsgroesse targetDim wird rausgezogen:
  //     firstHidden = targetDim * 2^n 
  // <=> firstHidden / targetDim = 2^n    | fuer alle targetDim != 0
  // <=> n = log2(firstHidden / targetDim)
  
  Net* net = new Net(numCopies);
  net->addLayer(new FullyConnectedLayer::FullyConnectedLayerArguments(width, height), true);
  for (int i=0; i < numReceptLayers; i++) { // create "sparse" layers
    net->addLayer(new ConvolutionLayer::ConvolutionLayerArguments(numKernels, fieldWidth, i==0 ? 1 : 2, shareWeights, i == 0 ? true : overlapping, false, shareBias), true);  // numKernels, kernelSize, stepsize, share weights, ueber die grenzen, summieren, share bias.
    net->layers[i+1]->connectLayer(net->layers[i]);
  }
  
  int size = net->layers[numReceptLayers]->numUnits / reductionBase;
  int threshold = secondLast > 0 ? secondLast : reductionBase * targetDim;
  
  vector<int> numUnits;
  while (size > threshold) {
    numUnits.push_back(size);
    size /= reductionBase;
  }
  numUnits.push_back(targetDim);
  for (int i=numUnits.size()-2; i >= 0; i--) {
    numUnits.push_back(numUnits[i]);
  }
  for (unsigned int i=0; i < numUnits.size(); i++) {
    net->addLayer(new FullyConnectedLayer::FullyConnectedLayerArguments(numUnits[i], 1), true);
    net->layers[net->getTopologyData().layerCount-1]->connectLayer(net->layers[net->getTopologyData().layerCount-2]);
  }
  net->addLayer(new FullyConnectedLayer::FullyConnectedLayerArguments(net->layers[numReceptLayers]->numCols, net->layers[numReceptLayers]->numRows), true);
  net->layers[net->getTopologyData().layerCount-1]->connectLayer(net->layers[net->getTopologyData().layerCount-2]);  
  
  for (int i=0; i < numReceptLayers; i++) {
    net->addLayer(new InvertedLayer::InvertedLayerArguments(net->layers[numReceptLayers-i-1]->numCols, net->layers[numReceptLayers-i-1]->numRows), true);  
    ((InvertedLayer*) net->layers[net->getTopologyData().layerCount-1])->createInvertedWeights(((IndividuallyConnectedLayer*)net->layers[numReceptLayers-i]));
    net->layers[net->getTopologyData().layerCount-1]->connectLayer(net->layers[net->getTopologyData().layerCount-2]);
  }
 
 // net.setUpdateFunc(0, params);
  net->initWeights(0, 0.1);  
    
  return net;
}


