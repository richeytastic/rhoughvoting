#pragma once
#ifndef HOUGHVOTING_RANDOM_FOREST_H
#define HOUGHVOTING_RANDOM_FOREST_H

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include "Forest.h"
using HoughVoting::Forest;
#include "RandomTree.h"
using HoughVoting::RandomTree;
using HoughVoting::PatchMatch;


namespace HoughVoting
{

class RandomForest : public Forest
{
public:
    typedef boost::shared_ptr<RandomForest> Ptr;

    // Load a RandomForest object from a forest save directory.
    static Ptr load( const string& forestDir);

    RandomForest( int numTrees, int minSamplesPerLeaf, int maxTreeDepth);
    virtual ~RandomForest();

    virtual int getNumTrees() const { return _trees.size();}

    // Set this before calling grow to cause this object to send regular updates
    // on training progress proportion complete to the parameter object.
    virtual void setTrainingProgressUpdater( ProgressDelegate* trainingUpdater);

    // Single-threaded - function blocks
    virtual void grow( const TrainSet& trainSet);

    virtual void printStats( ostream& os) const;

    // Single threaded.
    // Looks up PatchDescriptor against all trees and returns results
    // inside a vector of PatchMatch objects (already allocated by user).
    // treeMatches must be equal in length to the number of trees.
    virtual void doRegression( const PatchDescriptor::Ptr pd, vector<PatchMatch>& treeMatches) const;

    // On a single tree - index not checked!
    virtual float doTreeRegression( int treeIndex, const PatchDescriptor::Ptr pd, PatchSet** matches) const;

    // Save this forest to the given directory (trees within directory).
    // Trees save positive feature patch info if savePosFeatures == true (may be costly in space).
    virtual void save( const string& forestDir, bool savePosFeatures=false) const;

    virtual RandomTree* getTree( int treeIdx) { return _trees[treeIdx];}

    virtual cv::Mat_<float> makeProbMap( int maxPxlDim) const;

protected:
    vector<RandomTree*> _trees;

private:
    void saveMeta( const boost::filesystem::path&, const vector<string>& treefnames, bool savePosFeatures) const;
};  // end class

}   // end HoughVoting

#endif




