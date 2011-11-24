#ifndef _FULLYCONNECTEDLAYER_H_
#define _FULLYCONNECTEDLAYER_H_

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

#include <iostream>
#include <string>

#include "BasicLayerTypes.h"

namespace NPP2 {

  /** This is a basic layer where each neuron is connected to each neuron
   * of the previous layer (full connections). This layer is the standard
   * layer of NPP2 and is automatically used to realize standard 
   * feed-forward networks. */
  struct FullyConnectedLayer : public BasicLayerType {
    
    /** This layer-type does need no specific arguments besides the number of 
     * units in rows and columns that are needed by all layers. */
    class FullyConnectedLayerArguments : public BasicLayerArguments {
    public:
      /** constructor that arranges all units in a single column */
      FullyConnectedLayerArguments( int numUnits);
      FullyConnectedLayerArguments( int numCols, int numRows);
    };
    
    FTYPE* weights;  ///< vector holding the weights of all connections
    FTYPE* dEdw;     ///< vector holding the partial derivative "at" each weight
    
    FTYPE* delta;     ///< delta terms for each weight. calculated during update.
    FTYPE* variables; ///< holds termporary values (potentially) calculated by the update function for each individual weight. If and how these variables are used depends on the update function being used.
    
    int previousDim;  ///< dimension of the previous layer. The total number of connections is given by multiplying the previous layer's dimension with this layer's dimension.
    
  
    void forwardPass(FTYPE *input, int copy=0);  
    void backwardPass(FTYPE *dedo, int copy=0);
    void updateWeights(int numCopies=0);
    void connectLayer(const BasicLayerType* previousLayer);
    
    void initWeights(int mode, FTYPE range);
    
    void setUpdateFunction(const UpdateFunction* updateFunction);
    
    void writeToStream(std::ostream& out) const;
    void readFromStream(std::istream& in);
    
    /** constructs an empty layer */
    FullyConnectedLayer();
    /** constructs a layer with the given parameters */
    FullyConnectedLayer(Net* net, int layerId, int firstUnitId, int unitsPerRow, int numRows=1, int numCopies=0);
    /** fill a FullyConnectedLayerArguments - object and pass it to this 
     * constructor */
    FullyConnectedLayer(Net* net, int layerId, const LayerArguments*);
    virtual ~FullyConnectedLayer();
    
    virtual LayerArguments* getArguments() const;  
    
    virtual void copyWeights(const BasicLayerType* layer);  
  };

}

#endif
