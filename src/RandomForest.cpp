/************************************************************************
 * Copyright (C) 2017 Richard Palmer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#include "RandomForest.h"
using HoughVoting::RandomForest;
#include <cassert>
#include <ctime>
#include <iostream>
#include <iomanip>
using std::cerr;
using std::cout;
using std::endl;


RandomForest::RandomForest( int numTrees, int mspl, int md) : Forest(), _trees(numTrees, NULL)
{
    for ( int i = 0; i < numTrees; ++i)
        _trees[i] = new RandomTree( mspl, md);
}   // end ctor



RandomForest::~RandomForest()
{
    for ( size_t i = 0; i < _trees.size(); ++i)
        delete _trees[i];
}   // end dtor



void RandomForest::setTrainingProgressUpdater( ProgressDelegate* trainingUpdater)
{
    if ( trainingUpdater != NULL)
    {
        const int numTrees = _trees.size();
        for ( int i = 0; i < numTrees; ++i)
            _trees[i]->setTrainingProgressUpdater( trainingUpdater);
    }   // end for
}   // end setTrainingProgressUpdater



void RandomForest::grow( const TrainSet& trainSet)
{
    assert( trainSet.size() == 2);  // Only 2 classes allowed for now
    const int numTrees = _trees.size();
    for ( int i = 0; i < numTrees; ++i)
    {
        const int randInit = time(0) + i;
        _trees[i]->grow( &trainSet, randInit);   // Blocks
    }   // end for
}   // end grow



// public
void RandomForest::printStats( ostream& os) const
{
    const int numTrees = _trees.size();
    for ( int i = 0; i < numTrees; ++i)
    {
        // Print stats for each tree
        const vector<DepthStats>* dstats = _trees[i]->getStats();
        const int numFXs = dstats->at(0).featureCounts.size();

        os << "TREE " << std::setw(2) << i;
        for ( int f = 0; f < numFXs; ++f)
            os << " F" << std::setw(2) << std::setfill('0') << f;
        os << endl;
        os << std::setfill(' ');

        const int treeHeight = dstats->size();
        for ( int d = 0; d < treeHeight; ++d)
        {
            const int numNodes = dstats->at(d).numNodes;
            const int maxNodes = pow(2,d);  // Max nodes at this level
            const int pcntNodes = int(100 * float(numNodes)/maxNodes);
            const int leafNodes = dstats->at(d).numLeaves;
            os << "  -- D" << std::setw(2) << d << ") " << numNodes << "/" << maxNodes
               << " (" << pcntNodes << "%, " << leafNodes << " leaves)";
            if ( d < treeHeight - 1)
            {
                os << ":" << endl << "       ";
                // Print the feature utilisation
                const int numDecisions = numNodes - leafNodes;  // Number of decision (split) nodes at this level
                const vector<int>& fcs = dstats->at(d).featureCounts;
                for ( int f = 0; f < numFXs; ++f)
                {
                    const int pcntF = int(100 * float(fcs[f])/numDecisions);
                    os << " " << std::setw(3) << pcntF;
                }   // end for - feature counts
            }   // end if
            os << endl;
        }   // end for - tree levels
        os << endl;
    }   // end for - trees
}   // end printStats



void RandomForest::doRegression( const PatchDescriptor::Ptr pd, vector<PatchMatch>& matches) const
{
    const int numTrees = _trees.size();
    assert( matches.size() == numTrees);

    for ( int i = 0; i < numTrees; ++i)
    {
        PatchSet* treeMatches;
        float treeProb = _trees[i]->doRegression( pd, &treeMatches);
        if ( treeProb > 0)
            treeProb /= treeMatches->size(); // treeProb is now 1/L where L is the number of patches in the leaf

        matches[i].matches = treeMatches;
        matches[i].probability = treeProb;
    }   // end for
}   // end doRegression



float RandomForest::doTreeRegression( int treeIndx, const PatchDescriptor::Ptr pd, PatchSet** treeMatches) const
{
    float treeProb = _trees[treeIndx]->doRegression( pd, treeMatches);
    if ( treeProb > 0)
        treeProb /= (*treeMatches)->size();
    return treeProb;
}   // end doTreeRegression



#include <fstream>
void RandomForest::save( const string& saveDir, bool savePosFeatures) const
{
    // Create the directory if it doesn't already exist
    boost::filesystem::create_directory( boost::filesystem::path(saveDir));

    // Save out the trees into the directory
    const int numTrees = _trees.size();
    vector<string> treefnames;
    for ( int i = 0; i < numTrees; ++i)
    {
        std::ostringstream oss;
        oss << "T_" << i << ".tree";
        const string treefname = oss.str();
        boost::filesystem::path savePath = boost::filesystem::path(saveDir);
        savePath /= treefname;
        treefnames.push_back(treefname);

        cout << "Saving tree " << i << " to " << savePath.string() << endl;
        std::ofstream ofs( savePath.string().c_str());
        ofs << *_trees[i] << endl;
        ofs.close();
    }   // end for

    boost::filesystem::path metaPath = boost::filesystem::path(saveDir);
    metaPath /= "forest.meta";
    saveMeta( metaPath, treefnames, savePosFeatures);
}   // end save



void RandomForest::saveMeta( const boost::filesystem::path& savePath, const vector<string>& treefnames, bool svPsFt) const
{
    const string savefname = savePath.string();
    std::ofstream metafile( savefname.c_str());
    metafile << "NUM_TREES: " << _trees.size() << endl;
    metafile << "MAX_TREE_HEIGHT: " << _trees[0]->getMaxDepth() << endl;
    metafile << "MIN_SAMPLES_PER_LEAF: " << _trees[0]->getMinSamplesPerLeaf() << endl;
    metafile << "WITH_POS_FEATURES: " << (svPsFt ? "TRUE" : "FALSE") << endl;

    metafile << "PATCH_TYPE: " << (_fixedPatch ? "FIXED_PIXELS" : "SCALED_REAL") << endl;
    metafile << "PATCH_DIMS: " << _patchDims.width << " " << _patchDims.height << endl;
    metafile << "OFFSET_SCALING: " << (_offsetScaling ? "TRUE" : "FALSE") << endl;

    // Write out the feature specification strings
    if ( !_fxspecs.empty())
    {
        metafile << "NUM_FEATURE_SPECS: " << _fxspecs.size() << endl;
        for ( size_t i = 0; i < _fxspecs.size(); ++i)
            metafile << _fxspecs[i] << endl;
    }   // end if

    const int numTrees = treefnames.size();
    for ( int i = 0; i < numTrees; ++i)
        metafile << "TREE_FILE: " << treefnames[i] << endl;
    metafile << endl;
    // Write out the forest statistics
    this->printStats( metafile);
}   // end saveMeta



struct MetaData
{
    bool readFeatures;
    bool fixedPatch;
    cv::Size2f patchDims;
    bool offsetScaling;
    int numTrees;
    int minSamplesPerLeaf;
    int maxTreeDepth;
    vector<string> treefnames;
    vector<string> fxspecs;


    explicit MetaData( const string metaFile)
    {
        std::ifstream metafile( metaFile.c_str());
        string ln;
        while ( std::getline( metafile, ln))
        {
            std::istringstream iss(ln);
            string tok;
            iss >> tok;
            if ( tok == "NUM_TREES:")
                iss >> numTrees;
            else if ( tok == "MAX_TREE_HEIGHT:")
                iss >> maxTreeDepth;
            else if ( tok == "MIN_SAMPLES_PER_LEAF:")
                iss >> minSamplesPerLeaf;
            else if ( tok == "TREE_FILE:")
            {
                string treefname;
                iss >> treefname;
                treefnames.push_back(treefname);
            }   // end else if
            else if ( tok == "WITH_POS_FEATURES:")
            {
                string v;
                iss >> v;
                readFeatures = (v == "TRUE");
            }   // end else if
            else if ( tok == "PATCH_TYPE:")
            {
                string v;
                iss >> v;
                fixedPatch = (v == "FIXED_PIXELS");
            }   // end else if
            else if ( tok == "PATCH_DIMS:")
                iss >> patchDims.width >> patchDims.height;
            else if ( tok == "OFFSET_SCALING:")
            {
                string v;
                iss >> v;
                offsetScaling = (v == "TRUE");
            }   // end else if
            else if ( tok == "NUM_FEATURE_SPECS:")
            {
                int nfs;
                iss >> nfs;
                for ( int i = 0; metafile.good() && (i < nfs); ++i)
                {
                    string fxs;
                    std::getline( metafile, fxs);
                    fxspecs.push_back(fxs);
                }   // end for
            }   // end else if
            else
                break;  // Don't worry about anything else (e.g. tree stats)
        }   // end while

        metafile.close();
        assert( numTrees == treefnames.size());
    }   // end ctor
};  // end struct



// static
RandomForest::Ptr RandomForest::load( const string& forestDir)
{
    boost::filesystem::path metaPath = boost::filesystem::path( forestDir);
    metaPath /= "forest.meta";
    if ( !boost::filesystem::exists( metaPath))
        return RandomForest::Ptr();

    cout << "Loading Forest meta data" << endl;
    MetaData md( metaPath.string());
    RandomForest::Ptr forest( new RandomForest( md.numTrees, md.minSamplesPerLeaf, md.maxTreeDepth));

    // Load the trees into the forest
    for ( int i = 0; i < md.numTrees; ++i)
    {
        forest->_trees[i] = new RandomTree( md.minSamplesPerLeaf, md.maxTreeDepth);
        forest->_trees[i]->setReadWriteFeatures( md.readFeatures);
        boost::filesystem::path treepath = boost::filesystem::path( forestDir);
        treepath /= md.treefnames[i];
        if ( !boost::filesystem::exists( treepath))
            return RandomForest::Ptr();

        cout << "Loading Tree " << md.treefnames[i] << " ... ";
        std::ifstream ifs( treepath.string().c_str());
        ifs >> *forest->_trees[i];  // Load the tree
        cout << " done" << endl;
        ifs.close();
    }   // end for

    // Set the feature extraction spec strings
    forest->setFeatureExtractorSpecs( md.fxspecs, md.fixedPatch, md.patchDims, md.offsetScaling);
    return forest;
}   // end load



cv::Mat_<float> RandomForest::makeProbMap( int maxPxlDim) const
{
    // Collect all the tree leaf probability maps
    const int ntrees = _trees.size();
    cv::Mat_<float> pimg = _trees[0]->makeLeafProbMap( maxPxlDim);
    for ( int i = 1; i < ntrees; ++i)
        pimg += _trees[i]->makeLeafProbMap( maxPxlDim);

    int ksz = maxPxlDim / 10;
    if ( ksz % 2 == 0)
        ksz++;
    // Do a gaussian blur
    cv::GaussianBlur( pimg, pimg, cv::Size(ksz,ksz), 0, 0);

    const float sumProbs = cv::sum( pimg)[0];
    pimg /= sumProbs;

    return pimg;
}   // end makeProbMap
