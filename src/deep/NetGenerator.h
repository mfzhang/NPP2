#ifndef _npp2_netgenerator_h_
#define _npp2_netgenerator_h_

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

/*  NetGenerator.h
 *  Created by Sascha Lange on 25.11.08.
 */

namespace NPP2 {

  class Net;

  /** NetGenerators are used to construct complex networks with only the 
   * help of a few parameters. Derived classes may construct networks of
   * arbitrary complex structures. Implemented to wrap the construction of
   * various deep autoencoder networks. */
  class NetGenerator {
  public:
    
    /** desructs the generator. */
    virtual ~NetGenerator() {};
    
    /**
     * generates a net according to some internal definitions and algorithms. */
    virtual Net* generateNet(int targetDim) const=0;
    
  };

  /** NetGenerator for constructing deep autoencoders. Uses several locally
   * connected layers in the numRecptLayers outermost layers and 
   * FullyConnectedLayers afterwards. */
  class ConvolutionNetGenerator : public NetGenerator {
  public:
    
    /** Constructs a generator with the specified parameters. With the help
     * of the parameters its possible to construct a whole range of autoencoders:
     * autoencoders with and without receptive fields, with or without 
     * weight-sharing and with smaller or larger reduction factors from layer
     * to layer. More detailed explanations of the parameters can be found in 
     * the documentation of the ConvolutionLayer .
     *
     * \param widht         width of input layer (matches width of image)
     * \param height        height of input layer (matches height of image)
     * \param numCopies     number of internal copies of the weights. Limits the 
     *                      number of threads during training.
     * \param fieldWidth    width of the receptive fields / kernels
     * \param sharedWeights indicates whether or not to use shared weights
     * \param numKernels    number of kernels at each place of the input layer
     * \param reductionBase factor for reducing the size of the dimensions 
     * \param shareBias     indicates whether or not to share the bias weight
     * \param overlapping   should kernels overlap at the image borders?
     * \param numRecptLayers number of layers with sparse connection structure */
    ConvolutionNetGenerator(int width, int height, int numCopies, 
                            int fieldWidth = 9, bool shareWeights=true, 
                            int numKernels=1, double reductionBase=2., 
                            bool shareBias=false, bool overlapping=true, 
                            int numReceptLayers=2);
    
    /** destructs the generator. */
    ~ConvolutionNetGenerator();

    /** generate a net with the specified dimension of the code layer. */
    virtual Net* generateNet(int targetDim) const 
    { return generateNet(targetDim, 0); }
    
    /** generate a net with the specified dimensions of the code layer and
     * and layers directly adjacent to the code layer. This method can be used
     * to exactly specify the very last reduction from the last hidden layer
     * to the code layer. Experience has shown that this reduction should be
     * much larger than that of previous layers; typically using a factor
     * much larger than 2 (e.g. from 10 neurons directly to two neurons in the
     * code layer). Setting this "correctly" can have a large influence on the
     * results. */
    virtual Net* generateNet(int targetDim, int minSecondLast) const;
    
  protected:
    int width;           
    int height;
    int numCopies;
    int fieldWidth;
    bool shareWeights;
    int numKernels;
    double reductionBase;
    bool shareBias;
    bool overlapping;
    int numReceptLayers;
  };
  
}

#endif

