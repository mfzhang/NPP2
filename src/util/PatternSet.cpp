/******************************************************************************/
/* N++ : C++ class neural network simulator                                   */
/* (c) 1994 Martin Riedmiller, 2009, 2010 Sascha Lange                        */
/* last changed: 10.5.2010                                                    */
/* File: PatternSet.cpp                                                       */
/*       management (load, store) of patternfiles                             */
/******************************************************************************/

#include "PatternSet.h"
#include <fstream>
#include <iostream>

using namespace NPP2;
using namespace std;

PatternSet::PatternSet(void)
{
  pattern_count=0;
  input_count = target_count = 0;
  name = NULL;
  input = NULL;
  target = NULL;
}

int PatternSet::load_pattern(const string& filename)
{
  FILE *patf;
  char secondchance[MAX_STRING_LEN],s[MAX_STRING_LEN],part[MAX_STRING_LEN];
  char *value;
  int expecting;
  int i,j;
  long p;

  if((patf=fopen(filename.c_str(),"r")) == NULL){
    sprintf(secondchance,"%s.pat",filename.c_str());  /* try with extension .pat */
    if((patf=fopen(secondchance,"r")) == NULL){
      fprintf(stderr,"Can't open patternfile neither %s nor %s\n",filename.c_str(),secondchance);
      return (PAT_FILE_ERROR);
    }
  }
  
  /* Allocate pointer arrays of length max_no_of_pattern */
  name = new char* [MAX_NO_OF_PATTERN];
  if (name == 0){
    printf("Kein Speicherplatz bei Patternset!\n");
    exit(1);
  }
  input = new double* [MAX_NO_OF_PATTERN];
  if (input == 0){
    printf("Kein Speicherplatz bei Patternset!\n");
    exit(1);
  }
  target = new double* [MAX_NO_OF_PATTERN];
  if (target == 0){
    printf("Kein Speicherplatz bei Patternset!\n");
    exit(1);
  }

  expecting = 3;
  while ((expecting) && (fgets (s,MAX_STRING_LEN,patf) != NULL)){ 
    if (strncmp (s,"No.",2) == 0){
      sscanf(s,"%*s %*s %s",part);
      if (strncmp (part,"patterns",6) == 0){
	sscanf(s,"%*s %*s %*s %*s %s",part);
	/* pattern_count = atoi(part);  throw away information - don't trust */
	expecting --;
      }
      if (strncmp (part,"input",5) == 0){
	sscanf(s,"%*s %*s %*s %*s %*s %s",part);
	input_count = atoi(part);
	expecting --;
      }
      if (strncmp (part,"output",5) == 0){
	sscanf(s,"%*s %*s %*s %*s %*s %s",part);
	target_count = atoi(part);
	expecting --;
      }
    }	
  } /* end of reading header */
  
  p=0;
  while((fgets (s,MAX_STRING_LEN,patf) != NULL) && (p<MAX_NO_OF_PATTERN)){ 
    if(*s =='%'||*s ==' '||*s =='\n'); /* skip comments and empty lines */
    else{ /* new pattern read, allocate arrays */
      if (*s == '#'){ /* pattern has a patternname */
	name[p] = new char[LENPATNAME];
	if (name[p] == 0){
	  printf("Kein Speicherplatz bei Patternset!\n");
	  exit(1);
	}
	j=0;
	while(((name[p][j]=s[j])!=0) && j<LENPATNAME-1) j++;
	name[p][j] = 0;
      }
      else { /* read input and target */
	input[p] = new double [input_count];
	if (input[p] == 0){
	  printf("Kein Speicherplatz bei Patternset!\n");
	  exit(1);
	}
	target[p] = new double [target_count];
	if (target[p] == 0){
	  printf("Kein Speicherplatz bei Patternset!\n");
	  exit(1);
	}
	for(i=0,value=strtok(s, " \t");
	    (value!=NULL)&&i<input_count;value=strtok( NULL," \t" ),i++){
	  /* parse input line */
	  input[p][i] = (double)atof(value);
	} /* for */
	if (target_count>0){  /* read target pattern */
	  do{
	    char* ret = fgets (s,MAX_STRING_LEN,patf);
      if (!ret) {
        cerr << "Unexpected end of file while reading target pattern." << endl;
        exit(1);
      }
	  }
	  while ((*s =='%') || (*s =='#') || (*s ==' ') || (*s =='\n')); 
	  /* skip comments and empty lines */
	  for(i=0,value=strtok(s, " \t");
	      (value!=NULL)&&i<target_count;value=strtok( NULL," \t" ),i++){
	    /* parse target line */
	    target[p][i] = (double)atof(value);
	  }/* for */
	} /* if targets */
	p++;
      } /* else input/target pattern read */
    } /* end of else reading pattern */
  } /* while there's data in pattern file */
  pattern_count = p;
  fclose(patf);  /* Martin, wie konntest du das vergessen ???? */
  return(PAT_OK);
}

PatternSet::~PatternSet()
{
  long i;
  if (name){
    for(i=0;i < pattern_count;i++)
      if (name[i]) delete name[i];
    delete[] name;
  }
  if (input){
    for(i=0;i< pattern_count;i++)
      delete[] input[i];
    delete[] input;
  }
  if (target){
    for(i=0;i< pattern_count;i++)
      delete[] target[i];
    delete[] target;
  }
}

void PatternSet::print_pattern()
{
  int i;
  long p;

  printf("No. patterns %ld\nNo. inputs %d\nNo. outputs %d\n\n",
	 pattern_count,input_count,target_count);
  for(p=0;p<pattern_count;p++){
    if (name && name[p])
      printf("%s\n", name[p]);
    for(i=0;i<input_count;i++)
      printf("%5.3f  ",input[p][i]);
    printf("\n");
    for(i=0;i<target_count;i++)
      printf("%5.3f  ",target[p][i]);
    printf("\n\n");
  }
    
}


void PatternSet::save_pattern(const string& filename)
{
  int i;
  long p;
  
  ofstream out(filename.c_str());
  if (!out) {
    cerr << "Could not save pattern." << endl;
    return ;
  }
  
  // first generates a SNNS-like header
  out << "SNNS pattern definition file V1.4" << endl
      << "# generated by N++2 " << endl << endl;
  
  out << "No. of patterns     : " << pattern_count << endl 
      << "No. of input units  : " << input_count << endl
      << "No. of output units : " << target_count << endl << endl;

  // now print out all pattern
  for(p=0;p<pattern_count;p++){
    if (name && name[p])   
      out << name << endl;  // only prints out the name, if there is any
    else 
      out << "#" << p+1 << endl;
    for(i=0;i<input_count;i++)
      out << input[p][i] << "  ";
    out << endl;
    for(i=0;i<target_count;i++)
      out << target[p][i] << "  ";
    out << endl << endl;
  }
  
  out.close();
}
