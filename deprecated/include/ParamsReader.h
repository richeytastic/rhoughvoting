#pragma once
#ifndef HOUGHVOTING_PARAMS_READER_H
#define HOUGHVOTING_PARAMS_READER_H

#include <string>
using std::string;
#include <vector>
using std::vector;
#include <list>
using std::list;
#include <stdexcept>
#include <iostream>
#include <sstream>

#include <DataLoader.h>
using RFeatures::DataLoader;
using RFeatures::Instance;
//#include <PatchDescriptor.h>
//using RFeatures::PatchDescriptor;
#include <GallLempitskyFeatureExtractor.h>
using RFeatures::GallLempitskyFeatureExtractor;


namespace HoughVoting
{

class ParamsReader
{
public:
    class InvalidParameter : public std::runtime_error
    {public:
        InvalidParameter( const string msg) : std::runtime_error(msg) {}
    };  // end class

    // mode: 0 = generate/train features, 1 = test forest
    // Only parameters relating to the mode are read in.
    ParamsReader( const string& pfile, int mode) throw (InvalidParameter);
    ~ParamsReader();

    /***** FEATURE GENERATION *****/
    // Loads and returns the training instances - returns count.
    int loadTrainingData( int classId, vector<Instance>& instances) const;

    // Loads and returns the feature extractors - returns count.
    int getFeatureExtractors( vector<FeatureExtractor::Ptr>& fxs) const;
    int getPatchCount() const; // Number of patches per instance to extract
    string createSaveFeaturesFilename( int classId) const;

    /***** TEST DATA ******/
    // Loads and returns the test view or NULL if not found
    const View::Ptr getTestView() const;

    /**** TRAINING DATA ****/
    // Returns the number of classes loaded.
    //int getTrainingData( vector<int>& classLabels, vector<PatchDescriptor::Ptr>& pds, vector<int>& classCounts) const;
    // Returns the number of trees
    int getForestParams( int& minSamplesPerLeaf, int& maxHeight) const;

    // Returns the width and height of a scanning patch. If the
    // dimensions of the scanning patch are fixed pixels, the function
    // returns false. If the dimensions are real sized (metres), the
    // function returns true.
    bool getScanPatchParams( float& width, float& height) const;

    string getForestDir() const { return _forestDir;}

private:
    string _panosDir;   // Directory where panoramas stored (required for _trnFile)
    vector<string> _trnFiles;    // Training instance files for feature patch generation
    mutable list<string> _featNames;    // Generated names of features
    string _forestDir;  // Load/save directory for trained forest
    int _patchCount;    // Number of patches per instance to extract
    vector<FeatureExtractor::Ptr> _fxs;
    //vector<PatchDescriptor::Ptr> _pds;  // Patch descriptor sets loaded in order of params file
    vector<int> _labels;    // Corresponding class labels for patches
    vector<int> _ccounts;   // Counts for the respective classes
    int _numTrees, _minSamplesPerLeaf, _maxHeight;  // Forest params
    View::Ptr _testView;    // Test view
    cv::Size2f _patchSz; // Scanning patch size (fixed pixels or real size metres)
    bool _das;   // True if patch scanning adapts to depth, false if patch size is fixed

    void readFeatureExtractor( const string& featType, const string& imgType, std::istringstream& iss);
    //bool readPatchDescriptors( const string& pdfile);
    void readTestView( const string& panoFile, const string& viewFace);
};  // end class

}   // end namespace

#endif
