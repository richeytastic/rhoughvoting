/**
 * Used with Randomised Classification Trees to produce test parameter sets.
 */
#pragma once
#ifndef HOUGHVOTING_TEST_PARAM_GENERATOR_H
#define HOUGHVOTING_TEST_PARAM_GENERATOR_H

#include <vector>
using std::vector;

namespace HoughVoting
{

class TestParamGenerator
{
public:
    TestParamGenerator( int numFVs, int* fvRanges, int maxTests);

    int nextParams( int& i0, int& i1);

private:
    const int _maxTests;
    int _testIdx;
    bool _doRandom;

    vector<int> _fvRanges;  // Ranges for FVs

    int _fvId;         // Channel ID (which FV)
    vector<int> _i0;   // Index 0 for each FV
    vector<int> _i1;   // Index 1 for each FV

    bool generateRandom(int&, int&, int&) const;
    bool generateNext(int&, int&, int&);
};  // end class

}   // end namespace

#endif
