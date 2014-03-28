#ifndef _DEEPAUTOENCODER_H_
#define _DEEPAUTOENCODER_H_

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


/*  DeepAutoEncoder.h
 *  Created by Sascha Lange on 03.10.08.
 */

#include <iostream>
#include <vector>

namespace NPP2 {

  class Net;
  class PatternSet;

  /** Training parameters for the layer-wise pre-training phase. For
   * each layer that is trained during the pre-training phase, you can provide
   * an individual set of training parameters. */
  struct CascadeParams {
    int epochs;        ///< number of epochs to train the layer
    double deltaMax;   ///< RProp's deltaMax parameter to use with the layer
    double deltaStart; ///< RProp's delta0 parameter to use with the layer
    double decay;      ///< RProp's weight-decay paraemter to use with the layer
    
    /** Constructs a parameter set for layer-wise pre-training a single layer. */
    CascadeParams(double deltaMax=0.1, int epochs=50, 
                  double deltaStart=-.1, double decay=-.1) 
      : epochs(epochs), deltaMax(deltaMax), deltaStart(deltaStart), decay(decay) 
    {}
  };

  /** This class trains an encoder and a decoder for a given pattern set. This is
   * done by training the combined encoding-decoding MLP on the id operator and 
   * splitting the net in its encoding and decoding parts afterwards. The
   * auto-encoder is first trained using a layer-wise pretraining procedure, where
   * layers are trained individually. Afterwards, the pretrained weights are 
   * finetuned using the "standard" learning method. This class does not 
   * create a neural network, but wraps around an already constructed neural net.
   * This net is assumed to have an appropriate structure (autoencoder with
   * "symmetric" layer-sizes). An appropriate autoencoder can be constructed
   * using the ConvolutionNetGenerator. 
   *
   * \attention the values of the patterns are expected to be scaled to [0,1].
   *            for the training procedure only the input patterns are used (for
   *            both sides of the id-realizing net). Target patterns, if
   *            specified at all. */
  class DeepAutoEncoder {
  public:
    
    /** Constructs a DeepAutoEncoder around the given neural network. The 
     * network needs to have the appropriate structure. This class adds
     * only the facilities for the layer-wise pretraining and for deriving
     * encoder and decoder from the given net (by copying structure and weights). 
     * The parameter own determines, whether or not the given net should be
     * deleted when the DeepAutoEncoder is deconstructed. */
    DeepAutoEncoder(Net* net, bool own=true);
    
    /** Deconstructs the DeepAutoEncoder and deletes the given net, if 
     * ownership was handed to the DeepAutoEncoder instance. */
    ~DeepAutoEncoder();
    
    /** finetraining of the autoencoder network for numEpochs on the given
     * PatternSet.
     * \param patternSet  training pattern to use during training. Will only 
     *                    use the input pattern and ignore targets, as the net is 
     *                    trained on the id function.
     * \param numEpochs   number of epochs to train the network (trains at least
     *                    1 epoch)
     * \param out         output stream for printing-out human readable status
     * \param testpattern additional pattern set for testing the network during
     *                    training. The procedure will determine the test error
     *                    on this set after every 10 training epochs and print it 
     *                    out.
     * \param offset      added to the epoch for output purposes only. May be used
     *                    when continuing to train an already trained net in order
     *                    to produce a correct, plotable numbering of epochs.
     * \param numBatches  optional training in mini-batches. If >1, splits the
     *                    training set into numBatches parts and does a weight
     *                    update after processing a part.    
     * \return the training error (total sum of squares, tss). */
    double train(const PatternSet& patternSet, 
                 int numEpochs=0, std::ostream& out = std::cout,
                 const PatternSet* testpattern=0, int offset=0, 
                 int numBatches = 1); 
    
    /** applies layer-wise pre-training to the whole autoencoder. Starts with
     * the two outermost layers, puts them in a shallow autoencoder network and
     * trains them on the input pattern in the PatternSet. Afterwards, continues
     * with the next layers (working towards the center layer) and trains them
     * on reproducing the activations of the previous layers. The implementation
     * copies the weights to a newly constructed shallow autoencoder of the
     * same structure, trains that net and copies the resulting weights back
     * to the deep autoencoder. Besides the PatternSet, expects a vector that 
     * has a set of training parameters for each layer-wise training. If the
     * vector is empty, uses the same standard parameters for all layers. */
    void pretrain(const PatternSet&, const std::vector<CascadeParams>& params = std::vector<CascadeParams>(0));
    
    /** applies layer-wise pre-training to the whole autoencoder. Starts with
     * the two outermost layers, puts them in a shallow autoencoder network and
     * trains them on the input pattern in the PatternSet. Afterwards, continues
     * with the next layers (working towards the center layer) and trains them
     * on reproducing the activations of the previous layers. The implementation
     * copies the weights to a newly constructed shallow autoencoder of the
     * same structure, trains that net and copies the resulting weights back
     * to the deep autoencoder. Besides the PatternSet, expects a vector that 
     * has a set of training parameters for each layer-wise training. If the
     * vector is empty, uses the same standard parameters for all layers.
     * This version of the training procedure uses mini-batches. */    
    void pretrain(const PatternSet&, int numMiniBatches, const std::vector<CascadeParams>& params = std::vector<CascadeParams>(0));
    
    /** returns a newly constructed net that is only the encoder part of
     * the autoencoder. Copies structure and weights from the autoencoder. */
    Net* deriveEncoderNet() const;
    /** returns a newly constructed net that is only the decoder part of
     * the autoencoder. Copies structure and weights from the autoencoder. */
    Net* deriveDecoderNet() const;
    
    /** returns the full autoencoder net inside this wrapping class. The
     * net can be used as ever other net in NPP2. */
    Net* getNet() { return fullNet; }
    
  protected:
    Net* fullNet;    ///< the autoencoder net passed to the constructor
    bool own;        ///< flag indicating whether the instance owns the net and thus should delete it, when being deconstructed.
  };
  
}


#endif
