/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#include "tests/PIMBenchTestCases.h"
#include "gtest/gtest.h"

/*
 * PIMBenchTest:
 * micro benchmark for performance comparison between w/ PIM and w/o PIM
 */

using namespace DRAMSim;

TEST_F(PIMBenchFixture, gemv)
{
    cout << "input nc, n, num_centroids, out_features" << endl;

    uint32_t nc = 48;
    uint32_t n = 256;
    uint32_t num_centroids = 16;
    uint32_t out_features = 4096;

    cin >> nc;
    cin >> n;
    cin >> num_centroids;
    cin >> out_features;

    cout << "nc: " << nc << endl;
    cout << "n: " << n << endl;
    cout << "num_centroids: " << num_centroids << endl;
    cout << "out_features: " << out_features << endl;

    uint32_t batch = n;
    uint32_t dim_in = nc*num_centroids;
    uint32_t dim_out = out_features;

    setPIMBenchTestCase(KernelType::GEMV, dim_out, dim_in, batch);  // (KernelType, out_vec, in_vec)
    executeKernel();                                    // execute w/o PIM
    executePIMKernel();                                 // execute w/ PIM
    expectPIMBench(2.0);
}

TEST_F(PIMBenchFixture, mul)
{
    setPIMBenchTestCase(KernelType::MUL, 2 * 1024 * 1024, 2 * 1024 * 1024);
    executeKernel();
    executePIMKernel();
    expectPIMBench(2.0);
}

TEST_F(PIMBenchFixture, add)
{
    setPIMBenchTestCase(KernelType::ADD, 1024 * 1024, 1024 * 1024);
    executeKernel();
    executePIMKernel();
    expectPIMBench(2.0);
}

TEST_F(PIMBenchFixture, relu)
{
    setPIMBenchTestCase(KernelType::RELU, 4 * 1024 * 1024, 4 * 1024 * 1024);
    executeKernel();
    executePIMKernel();
    expectPIMBench(2.0);
}
