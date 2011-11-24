#ifndef _BASICLAYERTYPES_H_
#define _BASICLAYERTYPES_H_

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
#include "functions.h"
#include "LayerRegistry.h"

namespace NPP2 {
  
#ifdef __APPLE__
#pragma mark Basic defintions
#endif 

  enum LayerType { INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER }; ///< used to mark the type of the layers

  class Net;
  
  /** Data about the general layout of a specific layer. In NPP2 layers may be 
   * organized two-dimensionally in rows and columns. This is a feature that was
   * added to facilitate processing of two-dimensional image data. */
  class LayerTopology {
  public:
    int numUnits;   ///< total number of units (equals numRows * numCols)
    int numRows;    ///< number of rows in the layer (height)
    int numCols;    ///< number of columns in the layer (width)
    
    /** Constructor that practically puts all units in to a single, large row. */
    LayerTopology(int numUnits) 
      : numUnits(numUnits), numRows(1), numCols(numUnits) {}
    /** Constructor to organize the neurons in rows and columns. */
    LayerTopology(int cols, int rows) 
      : numUnits(cols*rows), numRows(rows), numCols(cols) {}
    /** Constructor for an empty (zero-initialized) layer. */
    LayerTopology() 
      : numUnits(0), numRows(0), numCols(0) {}
  };
  

 /** Class for collecting all the parameters that needed to be specified
  * during the construction of a layer of a specific type. Each layer type 
  * must offer its own corresponding Argument-Class that is derived from 
  * LayerArguments. With the help of these classes its possible to construct
  * layers in an "abstract" way (using a factory, without knowing the type
  * of the constructed layer) and nevertheless passing the right arguments. */
 class LayerArguments {
 public:  
   std::string identifer;         ///< string identifying the type of the layer
   
   /** This constructor should never be called directly, as it just produces
    * an argument object corresponding to an undefined layer type. */
   LayerArguments(const std::string& identifer="Undefined") : identifer(identifer) {} 
   /** Cleans up all the argumantes. */
   virtual ~LayerArguments() {}
 };
  
  
  
  
  
#ifdef __APPLE__
#pragma mark -
#pragma mark Basic Layer Type  (abstract base class)
#endif
  
  
  /** Abstract base class for all implementations of layer types. Defines all the
   * neurons together with basic facilities each layer must have (topological data, 
   * activation function, update function, etc.) but does not define the connection
   * structure (which is implementation dependent).
   *
   * General Conventions used in NPP2: 
   * - unit 0 in each layer is a bias unit and always on
   * - the normal units start at position 1 in the vector
   * - each layer has numUnits normal units 
   * - in each vector storing values for the units (neurons), the first numUnits+1 
   *   entries are for the 'real' unit. Afterwards there are numCopies copies of
   *   the values used for parallel processing. 
   * - there are copies for activations and derivatives, but never for the weights,
   *   as these are the same in each thread and are not touched (changed) by the
   *   threads but only after joining partial results from the different threads.
   */
  struct BasicLayerType {
    
    /** Correspoding class holding the arguments, every layer does expect
     * during construction: this are the number of units in columns and rows */
    class BasicLayerArguments : public LayerArguments {
    public:
      int numCols;          ///< number of units per column
      int numRows;          ///< number of units per row
          
      /** Constructs an argument object with the two necessary parameters. */
      BasicLayerArguments(int numCols, int numRows);
    };
    

    std::string identifer;  ///< string identifying the layer's type
    
    Net* net;               ///< points to the net, this layer is part-of
    
#ifdef __APPLE__
#pragma mark Layer-related properties
#endif
    int layerId;            ///< unique id of the layer, usually starts with 0
    int firstUnitId;        ///< number of the first unit in the layer; this is one larger than the previous' layers last neuron.
    int numUnits;           ///< total number of units in the layer
    int numRows;            ///< number of units per row
    int numCols;            ///< number of units per column
    int numCopies;          ///< number of copies of this layer (used during parallel processing)
    int numWeights;         ///< total number of weights (connections) TO this layer.
    
    
#ifdef __APPLE__
#pragma mark Unit-related properties
#endif
    FTYPE* dEdo;            ///< vector holding the partial derivatives for the output of each of the neurons. 
    FTYPE* dEdnet;          ///< vector holdign the partial derivatives for the net input of each neuron.
    FTYPE* out;             ///< activation (output) of each neuron.
    FTYPE* netin;           ///< net input to each neuron.
    int actId;              ///< identifies which activation function (linear, logistic) to use for all the units in this layer.
    
    FTYPE (*act_f)(FTYPE); ///< activation function
    FTYPE (*deriv_f)(FTYPE, FTYPE); ///< derivative of the activation function
    
    UpdateFunction* updateFunction; ///< update function (learning rule, e.g. RProp)
    
    
#ifdef __APPLE__
#pragma mark Accessing individual neurons
#endif    
    /** calculates the index into the arrays given the index of the unit and the
     * number of the copy to access */
    inline int calcIdFromIndex(int index, int copy=0) const;
    /** calculates the index into the arrays given the id of the unit and the 
     * number of the copy to access */
    inline int calcIndexFromId(int unitId, int copy=0) const; 
    
    /** given the index, calculates the x,y-coordinates of the unit and the
     * number of the copy, which this unit is in. */
    inline void calcXYFromIndex(int index, int *x, int *y, int *copy=0) const;
    /** given x,y-coordinates and the number of the copy to use, calculates
     * the index into the vectors. */
    inline int calcIndexFromXY(int x, int y, int copy=0) const;
    
    
#ifdef __APPLE__
#pragma mark Propagating, back-propagating and updating (learning)
#endif 
    /** propagates the given input through the layer, using the specified
     * copy. The net's propagation method will call this method with the
     * output of the previous layer. */
    virtual void forwardPass(FTYPE *input, int copy=0)=0;  
    /** back-propagates the given "input" through the layer, using the specified
     * copy. The net's backpropagation method will call this method with dedout
     * "received" from the subsequent layer */
    virtual void backwardPass(FTYPE *dedout, int copy=0)=0;
    /** updates the weights according to the caclulated error terms using an
     * appropriate learning method (e.g. backpropagation or RProp). */
    virtual void updateWeights(int numCopies=0)=0;
    

#ifdef __APPLE__
#pragma mark Initializing and handling the connection structure
#endif 
    /** connects this layer in an implementation-dependent way to the 
     * previous layer. */
    virtual void connectLayer(const BasicLayerType* previousLayer)=0;

    /** initializes the weights of the connections. 
     * \param mode  presently, must be 0 (uniform random-initialization)
     * \param range values will be from -range to range */
    virtual void initWeights(int mode, FTYPE range)=0;
    
    /** sets the update (learning) rule that is applied during the weight
     * update. */
    virtual void setUpdateFunction(const UpdateFunction* updateFunction)=0;
    /** sets the activation function that is used by all of this layer's neurons. 
     * \param actId either NPP_LINEAR or NPP_LOGISTIC . */ 
    virtual void setActivationFunction(int actId);
    
    /** serializes this layer to the given output stream. */
    virtual void writeToStream(std::ostream& out) const;
    /** de-serializes the layer from the given input stream. */
    virtual void readFromStream(std::istream& in);
    
    /** this method copies the weights from another layer of the same type
     * and identical structure. This is used during the "pre-training" of 
     * deep autoencoders, where layers are trained individually in another
     * (shallow) autoencoder network. */
    virtual void copyWeights(const BasicLayerType* layer)=0;  
    /** returns the type of this layer (INPUT_LAYER, OUTPUT_LAYER or
     * HIDDEN_LAYER) */
    int getLayerType() const;
    
    /** returns the arguments that have been used to construct this layer. */
    virtual LayerArguments* getArguments() const=0; // describe itself by returning arguments creating this layer.
    
    /** Base constructor using the Arguments-Objekt. */
    BasicLayerType(Net* net, int layerId, const LayerArguments*);
    /** Base constructor where the arguments are specified directly. */
    BasicLayerType(Net* net, int layerId, int firstUnitId, int unitsPerRow, int numRows=1, int numCopies=0);
    /** virtual destructor cleaning up all data structures connected to this 
     * layer. */
    virtual ~BasicLayerType();
    
  protected:
    virtual void initLayer(); ///< internal helper function that actually sets up all data structures. To be extended by derived classes.
  };
    
  
    
  inline std::ostream& operator<<(std::ostream& out, const BasicLayerType& lt) { lt.writeToStream(out); return out; }
  inline std::istream& operator>>(std::istream& in, BasicLayerType& lt) { lt.readFromStream(in); return in; }
  
  
  
  
  
  // /////// inlines ////////////////////////////////////////
  
  
  inline int BasicLayerType::calcIdFromIndex(int index, int copy) const
  { 
    index -= copy * (numUnits+1);
    return index == 0 ? 0 : (index-1) + firstUnitId;
  }
  inline int BasicLayerType::calcIndexFromId(int unitId, int copy) const
  { 
    return unitId == 0 ? numUnits * copy : (unitId - firstUnitId+1) + copy * (numUnits+1);
  }  
  
  inline void BasicLayerType::calcXYFromIndex(int index, int *x, int *y, int *copy) const
  { 
    if (copy) {
      *copy = index / (numUnits+1);
    }
    index = index % (numUnits+1);
    if (index == 0) {
      *x = -1;
      *y =  0;
    }
    else {
      *x = (index-1) % numCols;
      *y = (index-1) / numCols;
    }
  }
  
  inline int BasicLayerType::calcIndexFromXY(int x, int y, int copy) const
  {
    return (x < 0 && y <= 0) ? copy * (numUnits+1) : x+y*numCols+1 + copy * (numUnits+1);
  }
  
}

#endif
