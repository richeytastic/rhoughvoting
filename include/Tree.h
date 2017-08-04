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
#ifndef HOUGHVOTING_TREE_H
#define HOUGHVOTING_TREE_H

#include <ProgressDelegate.h>
using rlib::ProgressDelegate;
#include <vector>
using std::vector;
#include <boost/shared_ptr.hpp>
#include <PatchDescriptor.h>
using RFeatures::PatchDescriptor;

typedef vector<PatchDescriptor::Ptr> PatchSet;
typedef vector<PatchSet> TrainSet;


namespace HoughVoting
{

struct PatchMatch
{
    PatchSet* matches;
    float probability;
};  // end struct


class Tree
{
public:
    typedef boost::shared_ptr<Tree> Ptr;

    Tree();
    virtual ~Tree(){}

    void setTrainingProgressUpdater( ProgressDelegate* trainingUpdater);
    ProgressDelegate* getTrainingProgressUpdater() const { return _trainingUpdater;}

    // Outer vector is classes (currently only supports background - index 0,
    // and foreground - index 1 classes). PatchDescriptor objects must be stored
    // externally (e.g. in RandomForest) for the duration of use of this Tree object.
    virtual void grow( const TrainSet* trainSet, int randomInit) = 0;

    // Returns the foreground object probability and the list of matching patches (with offset vectors)
    virtual float doRegression( const PatchDescriptor::Ptr pd, PatchSet** matches) const = 0;

    virtual cv::Mat_<float> makeLeafProbMap( int maxDimPxls) const = 0;

protected:
    ProgressDelegate* _trainingUpdater;
};  // end class


}   // end namespace

#endif
