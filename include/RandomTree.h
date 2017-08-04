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

#pragma once
#ifndef HOUGHVOTING_RANDOM_TREE_H
#define HOUGHVOTING_RANDOM_TREE_H

#include "Tree.h"
using HoughVoting::Tree;
using HoughVoting::PatchMatch;
#include <iostream>
using std::ostream;
using std::istream;
#include <list>
using std::list;
#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>
#include <Random.h> // rlib


namespace HoughVoting
{

struct DepthStats
{
    int numNodes;   // All - including leaves
    int numLeaves;  // Number of leaf nodes at this level
    vector<int> featureCounts;  // Counts of optimal features used to split at this level
};  // end struct


class RandomTree : public Tree
{
public:
    RandomTree( int minSamplesPerLeaf, int maxDepth);
    virtual ~RandomTree();

    // Outer vector is classes (currently only supports background - index 0,
    // and foreground - index 1 classes). PatchDescriptor objects must be stored
    // externally (e.g. in RandomForest) for the duration of use of this RandomTree object.
    virtual void grow( const TrainSet* trainSet, int randomInit);

    // Returns the foreground object probability and the list of matching patches (with offset vectors)
    virtual float doRegression( const PatchDescriptor::Ptr pd, PatchSet** matches) const;

    virtual cv::Mat_<float> makeLeafProbMap( int maxDimPxls) const;

    // By default, on load and save, feature information is not parsed because it can be too
    // expensive to store and is not required for classification. Call this function with
    // enable set to true to enable feature data reading/writing (using input/output stream operators).
    void setReadWriteFeatures( bool enable=true);

    int getMinSamplesPerLeaf() const { return _minSamplesPerLeaf;}
    int getMaxDepth() const { return _maxDepth;}
    const vector<DepthStats>* getStats() const { return &_depthStats;}

private:
    struct Node;
    friend struct Node;

    const int _minSamplesPerLeaf;
    const int _maxDepth;
    int _totalFGs, _totalBGs;
    int _numFXs;
    vector<DepthStats> _depthStats; // Max possible nodes at each level (index) is 2^depth

    Node* _root;
    list<Node*> _leaves;

    bool _ioFeatures;

    rlib::Random _random;   // Random number generator

    int _totalPatchesParsed;
    void updateProgress( int nodeDepth, int numFGs, int numBGs, int featChannel);

    friend ostream& operator<<( ostream& os, const RandomTree&);
    friend istream& operator>>( istream& is, RandomTree&);
};  // end class


ostream& operator<<( ostream& os, const RandomTree&);
istream& operator>>( istream& is, RandomTree&);

}   // end namespace

#endif
