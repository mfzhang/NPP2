#ifndef _NPP2_layer_registry_h_
#define _NPP2_layer_registry_h_

/*****************************************************************************
 
 Copyright (c) 2009-2011 Sascha Lange
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

#include <string>
#include <vector>
#include "Registry.h"

namespace NPP2 {

class BasicLayerType;
class LayerArguments;
class Net;

/** Central registry of factory classes that are able to construct instances
    of a specific layer type. Implemented as singleton. It's named "factory",
    because from the outside it behaves like an abstract factory. On the inside,
    it's actually implemented as an registry, and uses the registered abstract
    factories to produce instances of a requested type. In order to register
    an additional layer type, use the Macro at the end of this file (examle:
    REGISTER_LAYERTYPE(NewLayerType, 'a new layer type where the given string
    matches the exact class name'); */
class LayerFactory : public AbstractRegistry {
public:
   
  /** List of parameters that must be known when constructing any layer-type. */
  class Arguments : public AbstractRegistry::Factory::Arguments {
  public:
    Net* net;                   ///< the neural net the new layer will be part-of
    int layerId;                ///< the id (enumerator) of the layer in the net
    const LayerArguments* args; ///< type-specific arguments. Defined together with each layer class.
    
    /** Constructor for creating an Argument-Object and "fill" it with the 
        given parameters. */
    Arguments(Net* net, int layerId, const std::string& nameOfType, const LayerArguments* args) 
      : AbstractRegistry::Factory::Arguments(nameOfType), net(net), layerId(layerId), args(args) {}
  };
  
  /** The factory is implemented as a singleton. The REGISTER_LAYERTYPE macro
      uses this method to get access to the registry. */
  static LayerFactory* getTheLayerFactory();

  /** Convenience method that wrapps looking-up the appropriate factory
      and then creating an instance into one single call. Retuns NULL, if 
      no factory is registered for the string passed in args.nameOfType. */
  BasicLayerType* create(const LayerFactory::Arguments* args) const;
  /** Convenience method that wrapps looking-up the appropriate factory
      and then creating an instance into one single call. Uses the default
      constructor / arguments to construct the instande. Retuns NULL, if 
      no factory is registered for the given string. */
  BasicLayerType* create(const std::string& nameOfType) const;

protected:
  static LayerFactory* the_layerFactory; ///< pointer to the singleton
};


/** This macro creates an implementation of the factory for the
    class "name" and adds it to the central LAYER factory. */
#define REGISTER_LAYERTYPE(name, desc)                                  \
class _layerfactory_##name : public AbstractRegistry::Factory {         \
public:                                                                 \
  _layerfactory_##name() {                                              \
    LayerFactory::getTheLayerFactory()->add(#name, desc, this);         \
  }                                                                     \
  void* create(const Arguments* args) const {                           \
    const LayerFactory::Arguments* largs = dynamic_cast<const LayerFactory::Arguments*>(args); \
    return new name (largs->net, largs->layerId, largs->args);          \
  }                                                                     \
  void* create() const { return new name(); }                           \
};                                                                      \
static _layerfactory_##name *_layerfactory_instance_##name =            \
  new _layerfactory_##name ();                                          

}

#endif
