#ifndef _NPP2_FUNCTIONS_H_
#define _NPP2_FUNCTIONS_H_

/*****************************************************************************
 
 Copyright (c) 1994, 2009-2011, Martin Riedmiller, Sascha Lange
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

/*
 *  File: functions.h
 *
 *  Created by Sascha Lange on 28.04.09.
 *  Based on n++ (c) 1994 Martin Riedmiller
 *  Last modified: 16.11.2011
 *
 *  Implementation of RProp.
 *  NOTE: Should be rather easy to port missing backprop from n++.
 */

#include <cmath>

#define FTYPE double          ///< use single or double precision floating point number?
#define CBLASPRECISION d      ///< precision character for calls of BLAS functions. Must match the choice of precision (s:float or d:double).

#define MAX_VARIABLES 10      ///< maximum number of variables for each connection between neurons
#define MAX_PARAMS 10         ///< maximum number of paramater to wheight update functions



#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

namespace NPP2 {
  

  /** logistic activation function */
  inline FTYPE logistic(FTYPE x)
  {
    if (x >= 16.0) x = 16.0;     /* avoid under/overflow */
    if (x <= - 16.0) x = -16.0;
    return( (FTYPE) (1.0 / (1.0 + exp( -x))) );
  }
  
  /** derivative of the logistic function */
  inline FTYPE logistic_deriv(FTYPE output, FTYPE netin)
  {
    return((1.0 - output) * output);  // hack: after forward pass output equals logistic(netin)
  }
    
  /** linear (id) function */
  inline FTYPE linear(FTYPE x)
  {
    return(x);
  }
  
  /** derivative of the linear function (slope = 1) */
  inline FTYPE linear_deriv(FTYPE, FTYPE)
  {
    return(1.0);
  }
    
  /** Abstract base class of all update functions. Basically the update
   * function initializes the "internal" variales that are attached to 
   * each neuron and then somehow updates the weight whenever called.*/
  class UpdateFunction {
  public:
    /** constructs an update function. */
    UpdateFunction();
    /** calculates a weight change (delta) given the present derivative of the
     * error in respect to the connection's weight and then applies the change
     * to the weight (weight += delta). Implementations may use the variables
     * attached to each weight for storing internal intermediate results and 
     * values (e.g. in order to realize a momentum term for each weight). */
    virtual void operator() (FTYPE* weight, FTYPE* delta, FTYPE* dEdw, FTYPE* variables)=0;
    /** set the parameter vector (semantics depend on particular implementation. */
    virtual void setParameters(const FTYPE* params)=0;
    /** get the parameter vector */
    virtual void getParameters(FTYPE* params) const=0;
    /** get the number of variables this update function uses. */
    inline   int getNumVariables() const { return numVariables; }
    /** called at the start of the procedure to give the update function a chance
     * to fill the variables at a neuron with the desired starting values */
    virtual void initVariables(FTYPE* variables) const=0;
    
    /** clones this update function by copying it's parameters. */
    virtual UpdateFunction* clone() const=0;
    
    /** cleans up all data structures associated with this update function. */
    virtual ~UpdateFunction() {}
  protected:
    int numVariables;  ///< how many "internal" variables does this update function use? These variables are stored at each weight.
  };

  /** Implements Riedmiller's Resilient propagation (RProp). */
  class RPROP : public UpdateFunction {
  public:
    virtual void operator() (FTYPE* weight, FTYPE* delta, FTYPE* dEdw, FTYPE* variables);
    virtual void setParameters(const FTYPE* params);
    virtual void getParameters(FTYPE* params) const;
    virtual void initVariables(FTYPE* variables) const;
    
    virtual UpdateFunction* clone() const;

    
    RPROP(FTYPE* params);
  protected:
    FTYPE delta0;
    FTYPE deltaMax;
    FTYPE weightDecay;
  };
  
  
  /** Abstract base class of all error functions. This allows NPP2 to use
   * different error functions in its net class. You can easily add your
   * own error function by deriving from this class. */
  class ErrorFunction {
  public:
    /** given an output value and a target value, return the error. */
    virtual FTYPE error(FTYPE output, FTYPE target) const=0;
    /** given an output value and a target value, return dedo. */
    virtual FTYPE deriv(FTYPE output, FTYPE target) const=0;
    
    virtual ~ErrorFunction() {}
  };
  
  /** The basic error type used in most cases. */
  class SquaredError : public ErrorFunction {
  public:
    virtual FTYPE error(FTYPE output, FTYPE target) const;
    virtual FTYPE deriv(FTYPE output, FTYPE target) const;    
  };
  
  /** Error function that is better suited for classification (0 / 1). */
  class UnimodalCrossEntropy : public ErrorFunction {
  public:
    virtual FTYPE error(FTYPE output, FTYPE target) const;
    virtual FTYPE deriv(FTYPE output, FTYPE target) const;    
  };
  
  /** Error function that is better suited for classification (0 / 1). */
  class MultimodalCrossEntropy : public ErrorFunction {
  public:
    virtual FTYPE error(FTYPE output, FTYPE target) const;
    virtual FTYPE deriv(FTYPE output, FTYPE target) const;    
  };
  
}

#endif

