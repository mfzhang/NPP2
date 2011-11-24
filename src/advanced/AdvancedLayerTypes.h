#ifndef _ADVANCEDLAYERTYPES_H_
#define _ADVANCEDLAYERTYPES_H_

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

#include <iostream>
#include <string>
#include <vector>
#include "functions.h"
#include "BasicLayerTypes.h"
#include "FullyConnectedLayer.h"
#include "npp2.h"


namespace NPP2 {  
  
  
  /** Special type of layer for implementing Cross-Entropy Error with Soft max 
   *  activation for multiple classes. Can be used only as output layer and must 
   *  be used with the Cross-Entropy Error. */
  struct MultimodalCrossEntropyOutputLayer : public FullyConnectedLayer {
    
    /** Layer-specific argument class. Does not need any additional arguments
     * besides those needed by the BasicLayer type. */
    class MultimodalCrossEntropyOutputLayerArguments : public FullyConnectedLayerArguments {
    public:
      MultimodalCrossEntropyOutputLayerArguments( int numUnits );
      MultimodalCrossEntropyOutputLayerArguments( int numCols, int numRows );
    };
    
    /** replaces the 'standard' propagation function. Replaces the separate 
     * calculation of the neurons' activations for each individual unit by a cross
     * entropy version that considers activations of all neurons to determine the
     * activation of each indvidual unit. */
    void forwardPass(FTYPE *input, int copy=0); 
    /** replaces the 'standard' back-propagation of errors in order to match
     * the cross-entropy activation in the forward pass. */
    void backwardPass(FTYPE *dedo, int copy=0);
    
    MultimodalCrossEntropyOutputLayer();
    MultimodalCrossEntropyOutputLayer(Net* net, int layerId, const LayerArguments* args);
    MultimodalCrossEntropyOutputLayer(Net* net, int layerId, int firstUnitId, int unitsPerRow, int numRows=1, int numCopies=0);
    virtual ~MultimodalCrossEntropyOutputLayer();
    
    virtual LayerArguments* getArguments() const;  
  };


  
  
  /** Layer with customized connection structure. Uses a 'n++-style' implementation
   * (link lists at each neuron). Since it doesn't use BLAS for propagating,
   * it should be preferably used with sparse connection structures. For fully
   * connected layers, use the FullyConnectedLayer for the same reason. The
   * implementation uses a list of connections that are represented by a 
   * Connection structure. */
  struct IndividuallyConnectedLayer : public BasicLayerType {
    
    /** Layer-specific argument class. Does not need any additional arguments
     * besides those needed by the BasicLayer type. */
    class IndividuallyConnectedLayerArguments : public BasicLayerArguments {
    public:
      IndividuallyConnectedLayerArguments(int numCols, int numRows);
    };
    
    /** Structure for representing a single connection between a neuron of the
     * previous layer to a neuron in this layer. Has "pointers" to both neurons
     * in the two layers and into the weight structures. */
    struct Connection {
      int from;     ///< index of the neuron in the previous layer
      int to;       ///< index of the neuron in this layer
      int index;    ///< index into the weight vectors
      /** Constructs a connection from neuron "from" to neuron "to" where its
       * weight is stored at "index" in the weight vector. */
      Connection(int from, int to, int index) : from(from), to(to), index(index) {}
      /** Constructs a zero-initialized connection. */
      Connection() : from(0), to(0), index(0) {}
    };
        
    std::vector<FTYPE> weights;  ///< holding the weights of all connections   
    std::vector<FTYPE> delta;    ///< holding the delta terms of all connections
    std::vector<FTYPE> variables;///< holding the variables of each connection
    std::vector<FTYPE> dEdw;     ///< holding the part. deriv. of all connections

    std::vector<Connection> connections; ///< list off all connection to this layer
    
    void forwardPass(FTYPE *input, int copy=0);  
    void backwardPass(FTYPE *dedo, int copy=0);
    void updateWeights(int numCopies=0);
    void connectLayer(const BasicLayerType* previousLayer);
    void initWeights(int mode, FTYPE range);

    /** adds another connection that connects neuron "from" to neuron "to"
     * to this layer. The index addresses the position in the weight vector at
     * which to store the new connections' weight. By letting several connections
     * point to the same entry in the vector, it's possible to realize 
     * weight-sharing. */
    void addConnection(int from, int to, int index);
    
    void setUpdateFunction(const UpdateFunction* updateFunction);
    
    void writeToStream(std::ostream& out) const;
    void readFromStream(std::istream& in);
    
    IndividuallyConnectedLayer();
    IndividuallyConnectedLayer(Net* net, int layerId, const LayerArguments* args);
    ~IndividuallyConnectedLayer();
    
    LayerArguments* getArguments() const; 
    
    virtual void copyWeights(const BasicLayerType* layer);  
  };
  
  
  
  
  /** This layer practically 'inverts' a given IndividuallyConnectedLayer. This
   * is done by creating the exact same connection structure but inverting the 
   * direction; if a connection was from a neuron i in the layer t to a neuron j 
   * in layer t+1, the new connection will be from neuron j in layer t to
   * neuron i in layer t+1. Thus, if the original layers had dimensions n 
   * and m, the dimensions of the layers connected by this class must have 
   * dimensions m and n. 
   * 
   * When this layer is applied to invert a layer that used weight-sharing,
   * the resulting layer will have the inverted connection structure, but will
   * NOT have shared weights. This is because the function calculated by a 
   * convolutional layer can not be inverted by the 'same' inverted layer when 
   * also using weight-sharing. Thus, connection that use the same weight will
   * use different weights in the resulting, InvertedLayer (additional entries
   * in the weight vector are created as needed). */
  struct InvertedLayer : public IndividuallyConnectedLayer {
   
    /** Layer-specific argument class. Does not need any additional arguments
     * besides those needed by the BasicLayer type. */
    class InvertedLayerArguments : public IndividuallyConnectedLayerArguments {
    public:
      InvertedLayerArguments(int numCols, int numRows);
    };
    
    InvertedLayer();
    InvertedLayer(Net* net, int layerId, const LayerArguments* args);
    
    /** Given an IndividuallyConnectedLayer copies its connection structure
     * to this layer using 'inverse' directions; connections running from
     * neuron i to j are copied to this layer running from j to i. Dimensions
     * of the given layer (and it's preceding layer) must match this layer's
     * dimension (in inverse order). */
    void createInvertedWeights(const IndividuallyConnectedLayer*);
  };
  
  
  
  
  /** This class realizes connection structures with small, locally connected
   * patches. It may be used to either realize receptive fields (local
   * connections that are independent of each other) or a convolutionary
   * layer (same local connections, but using shared-weights and thus
   * realizing a convolution with a kernel). This layer assumes a rather
   * sparse connection structure and does not utilize BLAS for speeding
   * up the calculations. */
  struct ConvolutionLayer : public IndividuallyConnectedLayer {
    
    /** Argument class wrapping the parameters used for constructing a 
     * ConvolutionalLayer. Needs information about the number and size
     * of kernels, about the space between kernels, about whether or not
     * to share weights (shared-weights: kernels, convolution, individual 
     * weights: receptive fields, independent processing) and related 
     * parameters. For a description of the arguments see the layer class. */
    class ConvolutionLayerArguments : public IndividuallyConnectedLayerArguments {
    public:
      int numKernels;
      int kernelSize;
      int stepsize;
      bool shareWeights;
      bool overBoundary;
      bool sum;
      bool shareBias;
      bool interleave;
      ConvolutionLayerArguments(int numCols, int numRows, int numKernels, int kernelSize, int stepsize, bool shareWeights=true, bool overBoundary=true, bool sum=true, bool shareBias= false, bool interleave=false);
      ConvolutionLayerArguments(int numKernels, int kernelSize, int stepsize, bool shareWeights=true, bool overBoundary=true, bool sum=true, bool shareBias=false, bool interleave=false);
    };
    
    int numKernels;    ///< number of different kernels resp. number of receptive fields at each place
    int kernelSize;    ///< size of kernel. Needs to be uneven. 
    int stepsize;      ///< space between kernels. A stepsize of "1" means "apply kernel at every (input) neuron", "2" means "apply kernel at every other neuron". 
    bool shareWeights; ///< indicates, whether or not to use weight-sharing. If it's true, this layer is a convolutional layer, if it's false, this layer will have individual receptive fields
    bool overBoundary; ///< indicates, whether or not kernels resp. receptive fields should 'overlap' the boundary of the input layer. If set to false, the center of the first receptive field will be (kernelSize-1) /2 neurons into the layer (both, x and y direction). If set to true, the center will be at the very first neuron. Parts of the kernle / receptive field that overlap the boundary are simply cut-off (thus, the overlapping fields are smaller). Setting of the parameter depends on the problem at hand. If unsure, set to false (more stable learning), especially, when using shared-weights.
    bool sum;          ///< whether or not the activations of the kernels / receptive fields at a particular place should be summed up (be connected to the same neuron). Actually, this should be almost always set to false, as summing up the kernels would be equivalent to just using one kernel.  
    bool shareBias;    ///< indicates, whether or not the bias weight should be also shared. The best setting depends on the problem at hand. In most experiments this should be set to false, in order to let the net adapted to locally differing lighting situations (when working with images). In most (all?) of LeCuns papers it remains unclear, whether bias weights have been shared or not.
    bool interleave;   ///< This parameter controls how to organize the neurons in this layer when numKernels > 1; should each kernel project to a separate "image" corresponding to the previous layer's "image"? Or should outputs of several kernels be interleaved with each other? 
    
    /** contructs the connection and weights structures according to the setting
     * of the parameters. */
    virtual void connectLayer(const BasicLayerType*);
    
    ConvolutionLayer();
    ConvolutionLayer(Net* net, int layerId, const LayerArguments* args);
    virtual ~ConvolutionLayer() {};
    
  protected:
    /** creates a locally connected patch centered at position x,y of the
     * previous layer. Connects these neurons to neuron "to", thus making
     * them part of "to's" receptive field. */
    void connectKernels(int x, int y, int to, int kernel);
  };
    
  
}

#endif
