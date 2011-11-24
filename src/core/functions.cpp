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
 *  File: functions.cpp
 *
 *  Created by Sascha Lange on 03.05.09.
 *  Based on n++h (c) 1994 Martin Riedmiller
 */

#include "functions.h"
#include <cassert>
#include <iostream>

using namespace std;
using namespace NPP2;

// these are some "magic" values partially obtained from Martin Riedmiller.
// these values should work in most cases with average results (but not in all!) 
#define UPDATE_VALUE 0.1
#define DELTA_MIN 1E-6
#define DELTA_MAX 50
#define ETAPLUS 1.2
#define ETAMINUS 0.5

UpdateFunction::UpdateFunction() : numVariables(0)
{}


RPROP::RPROP(FTYPE* params)
: UpdateFunction(), delta0(UPDATE_VALUE), deltaMax(DELTA_MAX), weightDecay(0.)
{
  numVariables = 2;
  setParameters(params);
}

void RPROP::setParameters(const FTYPE* params)
{
  delta0 = params[0];
  deltaMax = params[1];
  weightDecay = params[2];
  
  if (!delta0) delta0 = UPDATE_VALUE;
  if (!deltaMax) deltaMax = DELTA_MAX;
  if (delta0 > deltaMax) delta0 = deltaMax; // nonsense value?!
}

void RPROP::getParameters(FTYPE* params) const
{
  params[0] = delta0;
  params[1] = deltaMax;
  params[2] = weightDecay;
}

void RPROP::initVariables(FTYPE* variables) const
{
  variables[0] = delta0; // this holds the weight's individual step length that is adapted by rprop
  variables[1] = .1; // this is a term that reduces the step-length in the very first weight-update to 1/10th of the original length. this was thought to be useful when assembling individual (pre-trained) layers into a larger, deep network (last step of pre-training might be way to long for the first step in the deep net). Actually, reducing the length in such a way and only in the first step doesn't help much (if at all). I've nevertheless left it in here, in order to be able to reproduce my dissertation's results exactly.
}


UpdateFunction* RPROP::clone() const
{
  return new RPROP(*this);
}


void RPROP::operator() (FTYPE* weight, FTYPE* delta, FTYPE* dEdw, FTYPE* variables)
{
  register double direction, update_value, dEdwl;
  
  update_value = variables[0];
      
  dEdwl = *dEdw + weightDecay * *weight; // this implementation of weight decay has been debatted for quite some time in the group. either ignore or please ask a senior member for it's consequences ;-)
  direction = *delta * dEdwl;
      
  if(direction<0.0) {
    update_value = MIN(update_value * ETAPLUS,deltaMax);
    if(dEdwl>0.0) {
      *delta = - update_value;
    }
    else { // dEdw<0.0
      *delta = update_value;
    }
  }
  else if(direction>0.0){
    update_value = MAX(update_value * ETAMINUS,DELTA_MIN);
    *delta = 0.0;  // restart adaptation in next step
  }
  else{ /* direction == 0.0 */
    if(dEdwl>0.0)
      *delta = - update_value;
    else if(dEdwl<0.0)
      *delta = update_value;
    else
      *delta = (FTYPE) 0;
  }
  *weight += variables[1] * *delta;  // as stated above, variables[1] just reduces the step length in the very first step of the training procedure

  variables[0] = update_value;
  *dEdw = (FTYPE) 0;  /* important: clear dEdw !*/
  variables[1] = 1.;  // just use delta in all epochs after epoch 0
}


FTYPE SquaredError::error(FTYPE output, FTYPE target) const
{
  return (output-target)*(output-target);
}
FTYPE SquaredError::deriv(FTYPE output, FTYPE target) const
{
  return output - target;
}

FTYPE UnimodalCrossEntropy::error(FTYPE output, FTYPE target) const
{
  return - (target * log(output) + (1.-target) * log(1.-output));
}
FTYPE UnimodalCrossEntropy::deriv(FTYPE output, FTYPE target) const
{
  return - target/(output + 1e-15) + (1-target) / (1+1e-15-output);
}


FTYPE MultimodalCrossEntropy::error(FTYPE output, FTYPE target) const
{
  return target == 0. ? 0. : - (target * log(output / target));
}
FTYPE MultimodalCrossEntropy::deriv(FTYPE output, FTYPE target) const
{
  return output - target;
}


