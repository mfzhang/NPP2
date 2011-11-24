/*****************************************************************************
 
 Copyright (c) 2009, 2010, Sascha Lange
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


#include "Registry.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

#define ERROUT_REG( __x__ ) std::cerr << "#ERROR: " << __FILE__ << " in Line: " << __LINE__ << " in " << __PRETTY_FUNCTION__ << __x__ << "\n"


AbstractRegistry::AbstractRegistry() : entries()
{}

AbstractRegistry::~AbstractRegistry()
{
  for (unsigned int i=0; i < entries.size(); i++) {
    delete entries[i].factory;
  }
}

void AbstractRegistry::add(const std::string& name, const std::string& desc, Factory *factory) {
  Entry entry = { name, desc, factory };
  entries.push_back(entry);
}

AbstractRegistry::Factory* AbstractRegistry::lookup(const std::string& name) const
{
  for (unsigned int i=0; i < entries.size(); i++) {
    if (name == entries[i].name) {
      return entries[i].factory;
    }
  }
  return 0;  
}

int AbstractRegistry::listEntries(char* buf, int buf_size) const 
{
  buf[0] = '\0';
  int available = buf_size - 1;
  for (unsigned int i=0; i < entries.size(); i++) {
    strncat(buf, entries[i].name.c_str(), available);
    available -= strlen(entries[i].name.c_str());
    strncat(buf, "\n\t", available);
    available -= 2;
    strncat(buf, entries[i].desc.c_str(), available);
    available -= strlen(entries[i].desc.c_str());
    strncat(buf, "\n", available);
    available -= 1;
    if (available < 0) {
      return 1;
    }
  }
  buf[buf_size-1] = '\0'; // safety first
  return 0;
}