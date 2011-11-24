/******************************************************************************/
/* N++ : C++ class neural network simulator                                   */
/* (c) 1994 Martin Riedmiller, 2009-2011 Sascha Lange                         */
/* last changed: 16.11.2011                                                   */
/* File: PatternSet.h                                                         */
/* Purpose: PatternSet header file - to be included in application programs   */
/******************************************************************************/

#ifndef _NPP2_PATTERN_SET_H_
#define _NPP2_PATTERN_SET_H_

#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<string>

namespace NPP2 {

#define PAT_OK 0
#define PAT_FILE_ERROR -1
#define LENPATNAME 30
#define MAX_STRING_LEN 5000
#define MAX_NO_OF_PATTERN 10000000

  /** Represents a collection of input-output pairs that are used to 
   train and test a neural network. This very basic class for loading and storing
   patterns has been directly imported from n++. There are no facilities for
   programatically creating the lists or adding patterns (this has to be done 
   by hand), but the class 'owns' all lists and patterns and will free the
   memory during deconstruction. 
   
   \attention if created programatically, lists and patterns must be created 
              using the new [] operator (don't use malloc)! */
  class PatternSet{
  public:
    long pattern_count;     ///< number of pattern in the set
    int input_count;        ///< dimension of input vectors
    int target_count;       ///< dimension of target vectors
    char **name;            ///< list of names (each pattern can have a name)
    double **input;         ///< list of input patterns. Each entry is a vector. 
    double **target;        ///< list of taget patterns.
  
    /** Default constructor for constructing an empty pattern set. */
    PatternSet(void);
    /** Virtual Destructor that cleans up all the memory associated with the
     lists, names, and patterns. */
    virtual ~PatternSet();
  
    /** loads a pattern set from the given file. Expects a format compatible 
     to SNNS. */
    virtual int load_pattern(const std::string& filename);
    /** prints out all patterns in a human-readable form */
    virtual void print_pattern();
    /** saves the pattern set to the given file. Expects a format compatible 
     to SNNS. */
    virtual void save_pattern(const std::string& filename);
  };
  
}

#endif



