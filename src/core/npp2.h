#ifndef _NPP2_NPP2_H_
#define _NPP2_NPP2_H_

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

/*  Created by Sascha Lange on 28.04.09.
 *  Interface based on n++.h (c) 1994 Martin Riedmiller
 *  Last modified: 21.6.2011
 */

#include <pthread.h>
#include <iostream>
#include <vector>
#include "functions.h"
#include "NPPException.h"
#include "BasicLayerTypes.h"

namespace NPP2 {

  class PatternSet;
  
  /** Sturcture for representing errors. */
  struct Error {       
    double regrError;  ///< regression error
    double classError; ///< classification error, assuming unary encoded class labels
  };
    

  enum ActivationType { NPP_LOGISTIC=0, NPP_LINEAR };
  
  /** Parallel implementation of a multi-layer perceptron providing
   *  the core functionallity of N++2 
   *  including network construction, loading and saving, 
   *  propagating activations and errors as well as training
   *  the network. Differing from Martin Riedmiller's original N++ 
   *  this implementation
   *  uses a layer-oriented approach with an explicit representation of
   *  the layers. This allows faster propagation using BLAS-powered matrix
   *  operations instead of looping over individual connections.
   *  Whereas from the user this class has mainly the same interface as 
   *  the original N++ --- method names are "modernized" and slightly 
   *  changed in order to prevent accidental mis-use --- internally, this
   *  class controls n copies of the network structure in order to allow
   *  for massive parallelization during batch training procedures. The
   *  number of parallel threads to use is specified during network
   *  construction (parameter "numCopies") and must not be changed 
   *  afterwards during the whole lifecycle of the net. Copies and
   *  parallelization are completely transparent, as long as using
   *  the built-in methods for testing and training.
   */
  class Net {
  public:
   
#ifdef __APPLE__
#pragma mark -
#pragma mark Construction and destruction
#endif
/** \name Construction and destruction 
  @{ */
    
    /** constructs a neural network with numCopies copies of its internal 
     *  structure. This allows for a maximum of numCopies threads working in
     *  parallel during training and testing. This constructor only constructs
     *  the general network class but not the internal structur of neurons and
     *  weights. These have to be constructed calling further methods like
     *  createLayers / addLayers and connectLayers.
     *  \param numCopies number of copies to construct for parallel execution
     */
    Net(int numCopies = 0);
    ~Net();            ///< destructs the whole network including the connection structure
 
/*@}*/  
#ifdef __APPLE__
#pragma mark -
#pragma mark Accessing fundamental network information
#endif
/** \name Accessing fundamental network information 
  @{ */    
    
    int getNumCopies() const { return numCopies; } ///< returns the number of internal copies of the connection structure. This is an upper limit to the number of threads that can propagate / backpropagate in parallel.
    
    FTYPE* inVec;     ///< input vector of the neural net. May be filled with the input values before calling the forward propagation method.
    FTYPE* outVec;    ///< output vector of the neural net. This vector holds the output of the network after propagating the activations.
    
    /**
     * Structure holding user-relevant topologic information about the neural 
     * network including the number of layers and the size of the input layer 
     * and output layer.
     */
    struct TopologyData {
      int layerCount; ///< number of layers in neural network, including input layer and output layer.
      int inCount;    ///< number of neurons in input layer
      int outCount;   ///< number of neurons in output layer
    }; 
    
    const TopologyData& getTopologyData() const { return topoData; } ///< return the topology of the network (read only).
  
    std::vector<BasicLayerType*> layers;  ///< vector of pointers to the layers of the neural network.
 
/*@}*/  
#ifdef __APPLE__
#pragma mark -
#pragma mark Network creation and initialization
#endif
/** \name Network creation and initialization 
  @{ */    
    
    /**
     * creates a complete network with the specified number of layers using
     * the standard layer type (that is FullyConnectedLayer).
     * \param layers total number of layers including input and output layer
     * \param layerUnits array of the sizes of the individual layers, with its length matching the number of layers
     * \param numCopies number of copies to create. This argument overwrites the value passed to the constructor.
     */
    void createLayers(int numLayers, int *layerUnits, int numCopies=0) throw (NPPException);
    
    /**
     * creates a complete network with layers defined by the given specifications.
     * \param netSpecification vector of specifications of the individual layers
     * \param numCopies number of copies to create. This argument overwrites the value passed to the constructor.
     * \param cleanUpArgs convenience flag indicating whether or not the netSpecifications should be deleted afterwards
     */
    void createLayers(const std::vector<LayerArguments*>& netSpecification, int numCopies=0, bool cleanUpArgs=false) throw (NPPException);
    
    /**
     * adds a single layer on top of the existing network. The added layer
     * becomes the new output layer.
     * \param layer layer to be added to the network
     */
    void addLayer(BasicLayerType* layer);
    
    /**
     * adds a single layer on top of the existing network. The layer is
     * created according to the passed specification
     * \param args pointer to the specification of the layer
     * \param cleanUpArgs convenience flag indicating whether or not the netSpecifications should be deleted afterwards
     */
    void addLayer(LayerArguments* args, bool cleanUpArgs=false);
        
    /**
     * actually creates the connection structure between the neurons. Before
     * calling this method, the network must be assumed to not be connected,
     * although some layer types may connect themselves immediately after
     * insertation. After creating all layers by either using Net::createLayers
     * or Net::addLayer this method has to be called by the user before
     * the network can be used (e.g. by calling Net::forwardPass).
     */
    void connectLayers();
    
    /**
     * modifies the activation function of the neurons of an individual layer.
     * The standard activation function is the sigmoid.
     * \param layerNo number of layer that should be modified
     * \param actId id of the activation function
     */
    void setLayerActivationFunction(int layerNo, int actId);
    
    /**
     * sets the update function that is used during weight update.
     * \param typ id of the update function (default is NPP::RPOP)
     * \param params array of parameters used by the update function (at max MAX_PARAMS)
     */
    void setUpdateFunc(int typ, FTYPE *params);
    
    /**
     * initializes the weights of the connections
     * \param mode type of initialization
     * \param range range of the weights used during random initialization
     */
    void initWeights(int mode, FTYPE range);
 
/*@}*/  
#ifdef __APPLE__
#pragma mark -
#pragma mark Propagation and weight adaptation    
#endif
/** \name Propagation and weight adaptation 
  @{ */

    /**
     * propagates one pattern from the input layer to the output layer of the
     * neural network.
     * \param[in] inVec array of the input values applied to the input neurons. Must match the size of the input layer.
     * \param[out] outVec array where the network's output will be copied to. Must match the size of the output layer.
     * \param copy number of the internal copy of the network structure to be used for propagating. Must not be larger than the number of copies Net::getNumCopies.
     */
    void forwardPass(const FTYPE *inVec, FTYPE *outVec, int copy=0);
    
    /**
     * back-propagates the derivative of the error from the output layer to the input layer of the
     * neural network. Partial derivatives will be summed at each connection weight until Net::updateWeights is
     * called. 
     * \param[in] dedout array of the partial derivatives of the network error to be applied. Must match the size of the output layer.
     * \param[out] dedin array where the partial derivatives in respect to the network input will be copied to. Must match the size of the input layer.
     * \param copy number of the internal copy of the network structure to be used for propagating. Must not be larger than the number of copies Net::getNumCopies.
     */
    void backwardPass(const FTYPE *dedout, FTYPE *dedin, int copy=0);
    
    /**
     * updates the weights according to the selected update function and the summed partial derivatives of the error.
     * \param numCopies number of copies that have been used during propagation. The accumulated errors will be summed over all these copies.
     */
    void updateWeights(int numCopies=0);

/*@}*/ 
#ifdef __APPLE__
#pragma mark -
#pragma mark Pattern based learning and testing
#endif
/** \name Pattern based learning and testing 
  @{ */    
    
    /** trains the neural network on a pattern set for a single epoch using all available internal copies of the connection structure.
     * \param pattern training pattern
     * \param errorFunction errorFunction to optimize
     * \param id boolean specifying whether or not the network should be trained as an auto-encoder. In that case the inputs are also used as the targets.
     * \param numMiniBatches specifies the number of mini-batches (partition of the training patterns with multiple weight updates per epoch) to be used during training. 
     */
    double train(const PatternSet* pattern, const ErrorFunction* erorrFunction, bool id=false, int numMiniBatches=1);
    
    /** trains the neural network on a pattern set for a single epoch using a manually specified number of parallel threads.
     * \param pattern training pattern
     * \param threads number of parallel threads to use. Must not be larger than Net::getNumCopies.
     * \param id boolean specifying whether or not the network should be trained as an auto-encoder. In that case the inputs are also used as the targets.
     * \param errorFunction errorFunction to optimize
     * \param numMiniBatches specifies the number of mini-batches (partition of the training patterns with multiple weight updates per epoch) to be used during training. 
     */
    double train(const PatternSet* pattern, int threads=1, bool id=false, const ErrorFunction* erorrFunction = new SquaredError(), int numMiniBatches=1); ///< training function with an explicit number of threads

    /** tests the neural network on a pattern set using all available internal copies of the connection structure.
     * \param pattern testing pattern
     * \param errorFunction errorFunction to use to calculate the overal error on all testing patterns
     * \param id boolean specifying whether or not the network should be tested as an auto-encoder. In that case the inputs are also used as the targets.
     */
    Error test(const PatternSet* pattern, const ErrorFunction* erorrFunction, bool id=false);
    
    /** tests the neural network on a pattern set using a manually specified number of parallel threads.
     * \param pattern testing pattern
     * \param threads number of parallel threads to use. Must not be larger than Net::getNumCopies.
     * \param id boolean specifying whether or not the network should be tested as an auto-encoder. In that case the inputs are also used as the targets.
     * \param errorFunction errorFunction to use to calculate the overal error on all testing patterns
     */
    Error test(const PatternSet* pattern, int threads=1, bool id=false, const ErrorFunction* erorrFunction = new SquaredError()); ///< testing function with an explicit number of threads
  
/*@}*/ 
#ifdef __APPLE__
#pragma mark -
#pragma mark Loading and saving 
#endif 
/** \name Loading and saving  
  @{ */
    
    /** writes the neural network to a stream */
    void saveNet(std::ostream& out) const throw (NPPException);
    /** reads the neural network from a stream */
    void loadNet(std::istream& in) throw (NPPException);
    
    /** writes the neural network to a file with the specified name */
    void saveNet(const std::string& filename) const throw (NPPException);
    /** reads a neural network from a file with the specified name */
    void loadNet(const std::string& filename) throw (NPPException);
    
/*@}*/    
    
    friend class BasicLayerType;
    friend class DeepAutoEncoder;
    
  protected:
    TopologyData topoData;           ///< read-only information about the network's structure
    double updateParams[MAX_PARAMS]; ///< parameters of the weight update function
    UpdateFunction* updateFunction;  ///< pointer to the weight update function

#ifdef __APPLE__    
#pragma mark -
#pragma mark Internal, parallel implementation
#endif
    
    int numCopies;               ///< number of copies of the connection structure
    
    
    /** class for passing all the necessary information to and from a single
     *  worker during parallel training or testing.
     */
    struct WorkerData {
      Net* net;
      const ErrorFunction* errorFunction;
      pthread_t threadId;
      
      const PatternSet* pattern; ///< pointer to the complete pattern set
      int thread;                ///< number of the worker. used to find its copy of the network and its training patterns.
      int numThreads;            ///< total number of parallel threads used during training / testing
      bool trainId;
      int numMiniBatches;        ///< total number of mini batches to use during training
      int batch;                 ///< number of the present batch. Necessary, since parallel threads need to be synchronized during weight updates between the mini batches.
      
      double tss;                ///< total sum of squares on this thread's part of the data
      int countwrong;            ///< number of miss-classifications on this thread's part of the data
      
      WorkerData() {}            ///< default constructor
      WorkerData(Net* net, const ErrorFunction* errorFunction, const PatternSet* pattern, int thread, int numThreads, bool trainId, int batch=0, int numMiniBatches=1) ///< constructs and initializes the structure with all the necessary information
      : net(net), errorFunction(errorFunction), threadId(0), pattern(pattern), thread(thread), numThreads(numThreads), trainId(trainId), numMiniBatches(numMiniBatches), batch(batch), tss(0.), countwrong(0)
      {}
    };
    
    WorkerData* workerData;              ///< array of the data structures for each active worker
    static void* trainWorker(void* arg); ///< static hook to call the worker's training method during thread creation
    void trainWorker(WorkerData* arg);   ///< parallel training method executed by each worker

    static void* testWorker(void* arg);  ///< static hook to call the worker's testing method during thread creation
    void testWorker(WorkerData* arg);    ///< parallel testing method executed by each worker
    
    void deleteStructure();
    
/*  FUNCTIONALITY OF ORIGINAL N++ THAT HAS NOT BEEN PORTED, YET 
    FTYPE* scaled_in_vec;
    struct ScaleType {
      int scale_function;
      int position;
      FTYPE params[MAX_PARAMS];
    } *scale_list_in, *scale_list_out;
 
    void insert_scale_element(scale_typ **scale_list,int position,
                              int scale_function, double param1,double param2,double param3);
    void apply_scaling(double* data_vector, scale_typ *scale_list);
    void apply_backward_scaling(double* data_vector, scale_typ *scale_list); */
  };

}



#endif
