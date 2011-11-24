/*****************************************************************************
 
 Copyright (c) 1994, 2009-2011, Martin Riedmiller, Sascha Lange
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


/* N++2: Network creation and training demo adapted from:                    
 * n++ (c) 1994 Martin Riedmiller
 * created by Sascha Lange on 03.05.09.
 * last changed: 16.11.2011   
 */

#include <iostream>
#include "npp2.h"
#include "PatternSet.h"

#define LAYERS 3
#define HIDDEN_COUNT 5

using namespace std;
using namespace NPP2;

/** Demonstrates how to usethe PatternSet class and how to construct and train 
 * a basic neural net with four layers (input, output and two hidden layers). 
 * Examplary usage: ./bin/train_net examples/xor.pat 4 100            
 */
int main( int argc, char *argv[] )
{
  if(argc < 4){
    cerr << "Usage: " << argv[0] << " <Patterndatei> <Threads> <Epochen>" << endl;
    exit(0);
  }
  
  PatternSet pattern;
  pattern.load_pattern(argv[1]);    // load pattern 
  int numEpochs = atoi(argv[3]);    // number of training epochs
  int numThreads = atoi(argv[2]);   // number of parallel threads to use

  
  int topology[4] = {               // network with 4 layers
    pattern.input_count,            // size of input layer depends on the pattern
    10,                             // 10 hidden neurons in first layer
    10,                             // 10 hidden neurons in second layer
    pattern.target_count            // size of target layer depends on the patern
  };
  Net net;
  double param[3] = {               // RPROP parameters
    0.1,                            // delta 0
    0.8,                            // delta max
    0.0                             // weight-decay
  };
  
  net.createLayers(4, &topology[0], numThreads); // create 4 "standard" layers
  net.setUpdateFunc(0, &param[0]);  // use RPROP
  net.connectLayers();              // create the connections between the layers
  net.initWeights(0, .5);           // random initialization of from -.5 to .5
  
  for(int n=0; n < numEpochs; n++) {// loop for numEpochs 
    double tss = net.train(&pattern, new SquaredError());// train net for one epoch
    cerr << "Epoche " << n << ", tss: " << tss << endl; 
  }
  
  net.saveNet("trained.net");       // save the trained network
  
  // now pass all patterns through the net an print out the neural nets output.
  for (int n=0; n < pattern.pattern_count; n++) {
    net.forwardPass(pattern.input[n], net.outVec);
    cerr << "Pattern #" << n << ":";
    for (int i=0; i < pattern.input_count; i++) {
      cerr << " " << pattern.input[n][i];
    }
    cerr << " is mapped to";
    for (int i=0; i < net.getTopologyData().outCount; i++) {
      cerr << " " << net.outVec[i];
    }
    cerr << " (target:";
    for (int i=0; i < pattern.target_count; i++) {
      cerr << " " << pattern.target[n][i];
    }
    cerr << ")" << endl;
  }
  
  return 0;
}


