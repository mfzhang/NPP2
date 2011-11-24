/*****************************************************************************
 
 Copyright (c) 2009-2011, Sascha Lange, 
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

/*  ppmautoencoder.cpp
 *  Created by Sascha Lange on 27.05.09.
 *
 *  Purpose: Demonstrates encoding of images in a low-dimensional space. 
 *           Creates a collage of several reconstructions and writes it to disk.
 *
 *  Examplary usage:
 *
 *  > cd examples/carrera
 *  > ../../bin/ppmautoencode images.list 22 500    
 *
 *  (to run the training for 500 epochs with 22 threads on the images in 
 *  images.list)
 */

#include "npp2.h"
#include <iostream>
#include <fstream>
#include "DeepAutoEncoder.h"
#include "PatternSet.h"
#include "random.h"
#include "NetGenerator.h"
#include <cassert>


using namespace NPP2; 
using namespace std;

/** function for skipping white space */
static void skip(istream& in)
{
  char c;
  do {
    if (!in.get(c)) return;
  } while(isspace(c) || c=='\n' || c=='\r');
  if (c=='#') {                  // start of a comment
    do {                         // look for the end of the line
      if (!in.get(c)) return;
    } while(c!='\n');
    skip(in);                    // skip again at start of next line.
  }
  else                           // c was neither comment(#) nor space
    in.putback(c);               // so better put it back,
}

/** reads in a portable pixmap. can handle both, binary and ascii format. */
bool readGrayMap(const string filename, double** imageBuf, int* width, int* height)
{
  std::ifstream file(filename.c_str());
  if (!file) {
    return false;
  }
  
  // read header
  int type, maxCol;
  char c;
  
  skip(file);
  if (!file.get(c)) {
    file.close();
    return false;
  }
  if (c != 'P' && c != 'p') {
    file.close();
    return false;
  }
  
  file >> type;
  if (type != 5) {        // binary format?
    file.close();
    return false;
  }
  
  skip(file);
  file >> *width;
  skip(file);
  file >> *height;
  skip(file);
  file >> maxCol;         // 255 or smaller (bigger values not allowed)
  
  file.ignore(1);         // exactly one whitespace before start of binary data
  
  unsigned char buf[*width * *height];
  unsigned char* ptr = &buf[0]; 
  if (*imageBuf == 0) *imageBuf = new double [*width * *height];
  double* image = *imageBuf;
  
  double maxColD = maxCol;
  file.read(reinterpret_cast<char*>(&buf[0]), *width * *height);
  for (int i=*width**height; i > 0; i--) {
    *image++ = static_cast<unsigned char>(*ptr++) / maxColD;
  }
  
  file.close();
  
  return true;
}

/** reads in a bunch of images from the specified list of filenames. start and
 * step are used to specify which images to read and use for the pattern set
 * (e.g. every second image starting from image 0). This facility is used
 * to create training and pattern set from the same list of images. */
PatternSet* createPatternSetFromPPMS(const vector<string>& filenames, int start, int step, int* colsRet, int* rowsRet, int copies=0, double var=.1)
{
  const double MAX_DATA = 255.;
  
  double *image=0;
  int width, height;
  
  if (!readGrayMap(filenames[start], &image, &width, &height)) {
    cerr << "Could not read first image from " << filenames[start] << endl;
    return 0;
  }
  delete [] image;
    
  PatternSet* pattern = new PatternSet();
  pattern->pattern_count = (filenames.size()-start+(step-1)) / step;
  pattern->input_count = width*height;
  pattern->target_count = 2;
  pattern->input = new double* [pattern->pattern_count];
  pattern->target = new double* [pattern->pattern_count];
  
  
  unsigned int i; int pc=0;
  for (i=start, pc=0; i < filenames.size(); i+=step, pc++) {
    assert (pc < pattern->pattern_count);
    for (int c=0; c <= copies; c++) {
      pattern->input[pc*(copies+1)+c] = new double[pattern->input_count];
      pattern->target[pc*(copies+1)+c] = new double[pattern->target_count];
    }
    readGrayMap(filenames[i], &pattern->input[pc*(copies+1)], &width, &height);
    assert (width * height == pattern->input_count);
    
    for (int j=0; j < pattern->input_count; j++) {
      for (int c=1; c <= copies; c++) {
        double v = ((int)pattern->input[pc*(copies+1)][j]) / MAX_DATA + Tribots::nrandom(0., var);  // insert original and some noisy copies
        if (v < 0.) v = 0.; 
        if (v > 1.) v = 1.;
        pattern->input[pc*(copies+1)+c][j] = v;
      }
    }
  }
  assert (pc == pattern->pattern_count);
  *colsRet = width; *rowsRet = height;
  return pattern;
}


int main(int argc, char* argv[])
{
  
  if (argc != 4) {
    cerr << "usage: " << argv[0] << " image_list_file threads epochs" << endl;
    exit(1);
  }
  
  ifstream in(argv[1]);
  if (!in) {
    cerr << "Could not read images from " << argv[1] << endl;
    exit(1);
  }
  vector<string> filenames;
  while (in) {
    string filename;
    in >> filename;
    if (in) filenames.push_back(filename);
  }
  in.close();
  cerr << "Reading " << filenames.size() << " images." << endl;
  
  
  int cols, rows;
  int numCopies  = atoi(argv[2]);   // number of threads to use
  bool sharedWeights = true;
  int kernelSize = 5;
  int epochs = atoi(argv[3]);
  PatternSet* pattern = createPatternSetFromPPMS(filenames, 0, 2, &cols, &rows);
  PatternSet* patternTest = createPatternSetFromPPMS(filenames, 1, 2, &cols, &rows);
  
  pattern->save_pattern("images_train.pat");

  
  cerr << "Using " << pattern->pattern_count << " training and " 
       << patternTest->pattern_count 
       << " testing images with a size of " << cols << "x" << rows 
       << " pixels each." << endl;
  cerr << "Training for " << epochs << " epochs." << endl;

  FTYPE params[MAX_PARAMS] = {0.001, .5, 0.0001, 0., 0., 0., 0., 0., 0., 0.};

  ConvolutionNetGenerator generator(cols, rows, numCopies, kernelSize, sharedWeights, 1, 2., false /*bias*/, false /*overlapping at borders*/, 3);
  Net* net = generator.generateNet(3, 10);
  
   
  net->setUpdateFunc(0, params);
  net->initWeights(0, 0.1);
  

  net->saveNet("autoencoder_init.net");
  
  
  vector<CascadeParams> cparams;  // create parameter set for pre-training
  cparams.push_back(CascadeParams(.1,   50)); // 250));
  cparams.push_back(CascadeParams(.02,  100));
  cparams.push_back(CascadeParams(.02,  100));
  cparams.push_back(CascadeParams(.02,  100));
  cparams.push_back(CascadeParams(.01,  100));
  cparams.push_back(CascadeParams(.002, 100));
  cparams.push_back(CascadeParams(.001, 100));

  
  DeepAutoEncoder autoEncoder(net, true);
  autoEncoder.pretrain(*pattern, 5, cparams);
  
  net->saveNet("autoencoder_pretrained.net");
  
  for (int batch=0; batch < epochs/10; batch++) {
    char filename[500];

    
    /// Rekonstruktionen.
    
    int nc = 800 / cols;
    int nr = 600 /2 /rows ;
    
    while (nc*nr > patternTest->pattern_count) { // don't display more images than available
      if (nr > 2) nr /= 2;
      else nc /= 2;
    }
    int rowstride = nc * cols     + nc + 1; // (one pixel between the images
    int totalSize = rowstride * (rows+1) * nr * 2 + rowstride;   // top border
    
    
    FTYPE* images = new FTYPE[totalSize];

    for (int y=0; y < 2*nr+1; y++) {
      for (int x=0; x < nc*(cols+1)+1; x++) {
        images [ rowstride*(rows+1)*y + x] = y % 2 == 1 ? .3 : 1.;
      }
    }
    
    for (int y=0; y < 2*nr*(rows+1)+1; y++) {
      for (int x=0; x <= nc; x++) {
        images[ (rowstride)*y + x*(cols+1)] = 1.;
      }
    }
    
    
    for (int i=0; i < nc*nr; i++) {
      net->forwardPass((i < nc ? pattern : patternTest)->input[i], net->outVec);  // erste Reihe: Trainingspattern
      for (int r=0; r < rows; r++) {
        for (int c=0; c < cols; c++) {    
          images[ (rowstride)*((i/nc*2)  *(rows+1)+ r+1) + (i%nc)*(cols+1) + c + 1 ] = net->outVec[r*cols+c];
          images[ (rowstride)*((i/nc*2+1)*(rows+1)+ r+1) + (i%nc)*(cols+1) + c + 1 ] = (i < nc ? pattern : patternTest)->input[i][r*cols+c];
        }
      }
    }
    
    sprintf(filename, "reconstructions_%.4d.pgm", 10 * (batch));
    ofstream out (filename);
    out << "P5 " << ((cols+1)*nc+1) << " " << ((rows+1)*nr*2+1) << " 255" << endl;
    for (int j=0; j < totalSize; j++) {
      out.put((unsigned char)(images[j]*255));
    }
    out.close();
    delete [] images;
    
    
    
    autoEncoder.train(*pattern, 10, cout, patternTest, 10*batch); //, 10 - ((batch*10) / (epochs / 10))); 
    cerr << "Num mini batches: " <<   10 - ((batch*10) / (epochs / 10)) << endl;
    
    sprintf(filename, "autoencoder_%.4d.net", 10 * (batch+1));
    autoEncoder.getNet()->saveNet(filename);
  }
    
  return 0;
}
