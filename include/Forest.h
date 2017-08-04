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
#ifndef HOUGHVOTING_FOREST_H
#define HOUGHVOTING_FOREST_H

#include <ProgressDelegate.h>
using rlib::ProgressDelegate;

#include <iostream>
using std::ostream;
#include <vector>
using std::vector;
#include <string>
using std::string;
#include "Tree.h"
using HoughVoting::Tree;
using HoughVoting::PatchMatch;


namespace HoughVoting
{

class Forest
{
public:
    typedef boost::shared_ptr<Forest> Ptr;
    // For use by child classes - e.g. as:
    // static Ptr load( const string& forestDir);

    virtual ~Forest(){}

    virtual int getNumTrees() const = 0;

    // Set this before calling grow to cause this object to send regular updates
    // on training progress proportion complete to the parameter object.
    virtual void setTrainingProgressUpdater( ProgressDelegate* trainingUpdater) = 0;

    // The feature specification strings associated with the PatchDescriptors.
    // The feature specification strings are written out along with the forest
    // so that clients can create the correct feature extractor types for the forest.
    // fixedPatch: true for fixed pixel patches, false for real scaling size
    // patchDims: The pixel (integers) or real sized values for a patch's dimensions.
    // offsetScaling: Whether patch offsets are given as proportions of distance or as fixed pixel metrics.
    void setFeatureExtractorSpecs( const vector<string>& specs, bool fixedPatch, cv::Size2f patchDims, bool offsetScaling);

    // Populate specs with the feature extraction specification strings,
    // returning the number of strings added.
    int getFeatureExtractorSpecs( vector<string>& specs) const;

    bool useFixedPatches() const { return _fixedPatch;}
    cv::Size2f getPatchDims() const { return _patchDims;}
    bool useOffsetScaling() const { return _offsetScaling;}

    // Grow the forest - function blocks.
    virtual void grow( const TrainSet& trainSet) = 0;

    virtual void printStats( ostream& os) const = 0;

    // Function blocks.
    // Looks up PatchDescriptor against all trees and returns results
    // inside a vector of PatchMatch objects (already allocated by user).
    // treeMatches must be equal in length to the number of trees.
    virtual void doRegression( const PatchDescriptor::Ptr pd, vector<PatchMatch>& treeMatches) const = 0;

    // On a single tree - index not checked!
    virtual float doTreeRegression( int treeIndex, const PatchDescriptor::Ptr pd, PatchSet** matches) const = 0;

    // Save this forest to the given directory (trees within directory).
    // Trees save positive feature patch info if savePosFeatures == true (may be costly in space).
    virtual void save( const string& forestDir, bool savePosFeatures=false) const = 0;

    // Get the tree with given index
    virtual Tree* getTree( int treeIdx) = 0;

    // Return a probability map with values in [0,1]
    virtual cv::Mat_<float> makeProbMap( int maxPxlDim) const = 0;

protected:
    vector<string> _fxspecs;
    bool _fixedPatch;
    bool _offsetScaling;
    cv::Size2f _patchDims;
};  // end class

}   // end HoughVoting

#endif




