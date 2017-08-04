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

#include "Forest.h"
using HoughVoting::Forest;



void Forest::setFeatureExtractorSpecs( const vector<string>& specs, bool fixedPatch, cv::Size2f patchDims, bool offsetScaling)
{
    _fxspecs = specs;
    _fixedPatch = fixedPatch;
    _patchDims = patchDims;
    _offsetScaling = offsetScaling;
}   // end setFeatureExtractorSpecs



int Forest::getFeatureExtractorSpecs( vector<string>& specs) const
{
    specs.insert( specs.end(), _fxspecs.begin(), _fxspecs.end());
    return _fxspecs.size();
}   // end getFeatureExtractorSpecs
