#include "ParamsReader.h"
using HoughVoting::ParamsReader;
#include <fstream>
#include <cassert>
#include <sstream>


ParamsReader::ParamsReader( const string& pfile, int mode) throw (InvalidParameter)
{
    std::ifstream ifs( pfile.c_str(), std::ios_base::in);
    string ln, tok;
    while ( std::getline( ifs, ln))
    {
        // Empty lines and lines starting with the hash symbol are ignored
        if ( ln.empty() || ln[0] == '#')
            continue;

        std::istringstream iss( ln);
        iss >> tok;

        // Common parameters for training and testing
        if ( tok == "FOREST_DIR:")
            iss >> _forestDir;
        else if ( tok == "FEATURE:")
        {
            string featType, imgType;
            iss >> featType >> imgType;
            readFeatureExtractor( featType, imgType, iss);
        }   // end else if
        else if ( tok == "SQUARE_PATCH_DIMS:")
        {
            float dim;
            iss >> dim;
            _patchSz.width = dim;
            _patchSz.height = dim;
        }   // end if
        else if ( tok == "DEPTH_ADAPTIVE:")
        {
            string v;
            iss >> v;
            if ( v == "False" || v == "false" || v == "FALSE")
                _das = false;
            else if ( v == "True" || v == "true" || v == "TRUE")
                _das = true;
            else
                throw InvalidParameter( "Invalid true/false token for " + tok);
        }   // end else if

        // Parameters only relevant for patch feature generation and training
        if ( mode == 0)
        {
            if ( tok == "PANOS_DIR:")
                iss >> _panosDir;
            else if ( tok == "TRAIN_DATA:")
            {
                string tfile;
                iss >> tfile;
                _trnFiles.push_back(tfile);
            }   // end else if
            else if ( tok == "PATCH_COUNT:")
                iss >> _patchCount;
            else if ( tok == "NUM_TREES:")
                iss >> _numTrees;
            else if ( tok == "MIN_SAMPLES_PER_LEAF:")
                iss >> _minSamplesPerLeaf;
            else if ( tok == "MAX_TREE_HEIGHT:")
                iss >> _maxHeight;
        }   // end else if
        else if ( mode == 1)    // Parameters only relevant for testing
        {
            if ( tok == "TEST_PANO:")
            {
                string panoFile, viewFace;
                iss >> panoFile >> viewFace;
                readTestView( panoFile, viewFace);
            }   // end else if
        }   // end else if
    }   // end while
}   // end ctor



ParamsReader::~ParamsReader()
{}   // end dtor



// public
int ParamsReader::loadTrainingData( int classId, vector<Instance>& instances) const
{
    if ( classId < 0 || classId >= _trnFiles.size())
        throw InvalidParameter( "ParamsReader::loadTrainingData called with out-of-range class ID");
    DataLoader( _panosDir).loadInstances( _trnFiles[classId], instances);
    const int num = instances.size();
    std::cerr << "Loaded " << num << " instances from " << _trnFiles[classId] << std::endl;
    return num;
}   // end loadTrainingData



// public
int ParamsReader::getFeatureExtractors( vector<FeatureExtractor::Ptr>& fxs) const
{
    fxs.insert( fxs.end(), _fxs.begin(), _fxs.end());
    return _fxs.size();
}   // end int



// public
int ParamsReader::getPatchCount() const
{
    return _patchCount;
}   // end getPatchCount



// public
string ParamsReader::createSaveFeaturesFilename( int classId) const
{
    if ( classId < 0 || classId >= _trnFiles.size())
        throw InvalidParameter( "ParamsReader::createSaveFeaturesFilename called with out-of-range class ID");

    // Otherwise construct from other details
    string dataName = _trnFiles[classId].substr(_trnFiles[classId].find_last_of('/')+1);    // Filename - not whole path
    dataName = dataName.substr(0, dataName.find_last_of('.')); // Remove any extension

    std::ostringstream oss;
    oss << dataName << "_";
    size_t lindex = 0;
    for ( list<string>::const_iterator i = _featNames.begin(); i != _featNames.end(); ++i)
    {
        oss << *i;
        if ( lindex++ < (_featNames.size() - 1))
            oss << "+";
    }   // end for

    // Scaled or fixed size patch extraction?
    if ( _das)
        oss << "_SCALED" << _patchSz.width;
    else
        oss << "_FIXED" << _patchSz.width;

    oss << ".pf";   // pf = Patch Feature
    return oss.str();
}   // end createSaveFeaturesFilename



// public
const View::Ptr ParamsReader::getTestView() const
{
    return _testView;
}   // end getTestView



/*
// public
int ParamsReader::getTrainingData( vector<int>& labels, vector<PatchDescriptor::Ptr>& pds, vector<int>& ccounts) const
{
    assert( !_labels.empty());
    labels.insert( labels.end(), _labels.begin(), _labels.end());
    pds.insert( pds.end(), _pds.begin(), _pds.end());
    ccounts.insert( ccounts.end(), _ccounts.begin(), _ccounts.end());
    return *_labels.rbegin();
}   // end getTrainingData
*/



// public
int ParamsReader::getForestParams( int& mspl, int& mh) const
{
    mspl = _minSamplesPerLeaf;
    mh = _maxHeight;
    return _numTrees;
}   // end getForestParams



// public
bool ParamsReader::getScanPatchParams( float& width, float& height) const
{
    width = _patchSz.width;
    height = _patchSz.height;
    return _das;
}   // end getScanPatchParams



// private
void ParamsReader::readFeatureExtractor( const string& featType, const string& imgType, std::istringstream& iss)
{
    if ( imgType != "DEPTH" && imgType != "COLOUR")
        throw InvalidParameter( "Unknown parameter: " + imgType);

    const bool useDepth = imgType == "DEPTH";
    if ( featType == "GL")
    {
        cv::Size featurePatch;
        int numBins;
        iss >> featurePatch.width >> numBins;
        featurePatch.height = featurePatch.width;   // Square dims
        FeatureExtractor::Ptr fx( new GallLempitskyFeatureExtractor( useDepth, featurePatch, numBins));
        _fxs.push_back(fx);
        std::cerr << "Created " << imgType << " GallLempitskyFeatureExtractor with " << numBins << " HOG bins encoding as "
            << featurePatch.width << " x " << featurePatch.height << " patch features" << std::endl;

        // Write a string description of the feature details for the feature savefile name
        std::ostringstream oss;
        oss << "GL-" << imgType << "-" << numBins << "-" << featurePatch.width;
        _featNames.push_back( oss.str());
    }   // end if
    else
        throw InvalidParameter( "Only GL feature type currently supported");
}   // end readFeatureExtractor


/*
// private
bool ParamsReader::readPatchDescriptors( const string& pdfile)
{
    bool readOkay = false;
    const int numLoaded = PatchDescriptor::load( pdfile, _pds);
    if ( numLoaded > 0)
    {
        if ( _pds[0]->getNumFeatureVectors() == 0)
            throw InvalidParameter( "PatchDescriptor read in with ZERO feature vectors!");
        else
        {
            int classLabel = 0;
            if ( !_labels.empty())
                classLabel = _labels[_labels.size()-1] + 1; // Next class label one more than last
            _labels.insert( _labels.end(), numLoaded, classLabel);
            std::cerr << "Read " << numLoaded << " patch descriptors from " << pdfile << std::endl;
            _ccounts.push_back( numLoaded);
            readOkay = true;
        }   // end else
    }   // end else
    return readOkay;
}   // end readPatchDescriptors
*/



void ParamsReader::readTestView( const string& panoFile, const string& viewFace)
{
    using RFeatures::PanoramaReader;
    Panorama::Ptr pano = PanoramaReader::read( panoFile);
    if ( pano == NULL)
        throw InvalidParameter( "Invalid panorama file " + panoFile);

    if ( viewFace == "FRONT")
        _testView = PanoramaReader::getViewFace( pano, 0);
    else if ( viewFace == "LEFT")
        _testView = PanoramaReader::getViewFace( pano, 1);
    else if ( viewFace == "REAR")
        _testView = PanoramaReader::getViewFace( pano, 2);
    else if ( viewFace == "RIGHT")
        _testView = PanoramaReader::getViewFace( pano, 3);
    else
        throw InvalidParameter( "Invalid view face");

    if ( _testView == NULL)
        throw InvalidParameter( "Unable to set view face from panorama");

    std::cerr << "Read in " << viewFace << " view face from " << panoFile << std::endl;
}   // end readTestView
