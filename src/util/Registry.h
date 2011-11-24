#ifndef _NPP2_registry_h_
#define _NPP2_registry_h_

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

#include <string>
#include <vector>


/** Central registry of factory classes that are able to construct instances
    of a specific type. */
class AbstractRegistry{
public:
  
  
  /** This abstract base class declares an abstract factory, that is able
      to construct instances of one specific class. Each class needs a 
      different implementation of the abstract factory. */
  class Factory {  
  public:
    
    /** This class is used to wrap the class-specific parameters that are
        used during construction of an instance of the class. Different
        class-specific implementations of the abstract factory may come
        with their own, class-specific Arguments-implementation. */
    class Arguments {
    public:
      const std::string& nameOfType;  ///< name of the class to construct. This string is used to look up the appropriate factory in the registry.
      /** Default constructor that does not need any parameters besides the name
          of the class to construct. */
      Arguments(const std::string& nameOfType) : nameOfType(nameOfType) {} 
      /** Virtual destructor that should clean up all memory associated with the
          arguments. */
      virtual ~Arguments() {}
    };  
    
    /** Virtual destructor. Should be implemented in each derived class to clean
        up associated memory. */
    virtual ~Factory() {}
    /** Constructs an instance passing the given arguments. */
    virtual void* create(const Arguments* args) const=0;
    /** Constructs an instance using the default constructor / arguments. */
    virtual void* create() const=0;
  };    

  /** Default destructor cleaning up the registry. */
  virtual ~AbstractRegistry();
  
  /** add a factory for the class "name". The given factory is owned
      by the registry and thus will be cleaned-up by the factory. */
  virtual void add(const std::string& name, const std::string& desc, Factory* factory);
  /** lookup a factory for the requested class "name". Returns NULL if there's
      no class registered for the given string. */
  virtual Factory* lookup(const std::string& name) const;

  /** prints a human readable description of the registered factories into
      the given buffer. */
  virtual int listEntries(char* buf, int buf_size) const;

protected:

  /** Default constructor is hidden in order to prevent direct construction
      of this abstract base class. Derived classes are allowed to (and should)
      use this default constructor. */
  AbstractRegistry();              // Abstract class, can not be created
  
  /** Definition of one entry in the Registry. */
  struct Entry {                   // mapping name -> factory
    std::string name; ///< name of the type, used to lookup the appropriate factory
    std::string desc; ///< human-readable description of the type
    Factory* factory; ///< factory for constructing this type
  };

  std::vector<Entry> entries; ///< list of registered factories
};

#endif
