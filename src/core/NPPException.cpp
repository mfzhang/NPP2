/*****************************************************************************
 
 Copyright (c) 2010, 2011 Sascha Lange, backtrace method adapted from tribots
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

#include "NPPException.h"
#if 0
#include <execinfo.h>
#endif

using namespace NPP2;

NPPException::NPPException (const std::string& description) {
  this->description = std::string ("NPPException: ")+description;
#if 0
  btraceSize = ::backtrace (btrace, 50);
#endif
}

std::string NPPException::what () throw () {
  return description;
}

const std::vector<std::string>& NPPException::backtrace () throw () {
#if 0
  if (btraceText.size()==0) {
    btraceText.resize (btraceSize);
    char** btraceSymbols = ::backtrace_symbols(btrace, btraceSize);
    for (int i=0; i<btraceSize; i++)
      btraceText[i] = btracSymbols[i];
    free (btraceSymbols);
  }
#endif
  return btraceText;
}
