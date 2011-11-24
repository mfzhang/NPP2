#ifndef _npp2_nppexception_h_
#define _npp2_nppexception_h_

/*****************************************************************************
 
 Copyright (c) 2010, 2011 Sascha Lange, adapted from tribots
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

#include <exception>
#include <vector>
#include <string>

namespace NPP2 {
  
  /** NPP2-specific exception, copied from the tribots code. */
  class NPPException : public std::exception {
  protected:
    std::string description;                ///< verbal descritpion of the error that occured
    void* btrace [50];                      ///< array for backtrace
    int btraceSize;                         ///< size of backtrace
    std::vector<std::string> btraceText;    ///< backtrace as string
    
  public:
    /** constructor that expects a description of the error
     *  \param description verbal description of the error
     */
    NPPException (const std::string& description);
    
    /** destructor */
    virtual ~NPPException () throw() {;}
    
    /** returns the verbal description of the error  */
    virtual std::string what () throw();
    
    /** returns the backtracke */
    virtual const std::vector<std::string>& backtrace () throw ();
  };
  
  
}

#endif