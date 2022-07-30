import os
import sys
import filecmp

import numpy as np


def strings2array(strs):
    return np.array([[float(i) for i in j.strip().split()] for j in strs])


def read_p1a_chart(file):
    lines = open(file).readlines()
    utt2a_probs = strings2array(lines[4:73])
    utt2a_arcs = strings2array(lines[77:146])
    assert utt2a_probs.shape == (69, 123)
    assert utt2a_arcs.shape == (69, 123)
    return utt2a_probs, utt2a_arcs


def read_gmm(file):
    lines = open(file).readlines()
    gmm = strings2array(lines[322:424])
    assert gmm.shape == (102, 24)
    return gmm


def read_p3a_chart(file):
    lines = open(file).readlines()
    utt2a_forw = strings2array(lines[4:73])
    utt2a_back = strings2array(lines[77:146])
    utt2a_post = strings2array(lines[150:218])
    assert utt2a_forw.shape == (69, 20)
    assert utt2a_back.shape == (69, 20)
    assert utt2a_post.shape == (68, 102)
    return utt2a_forw, utt2a_back, utt2a_post

if __name__ == '__main__':
    output_dir = sys.argv[1]
    os.system('make clean -C {}/src'.format(output_dir))
    os.system('make -C {}/src'.format(output_dir))

    print('Checking p1a...')
    os.system('cd {} && bash lab2_p1a.sh > p1a.out'.format(output_dir))
    assert filecmp.cmp('{}/p1a.out'.format(output_dir), 'out/p1a.out')
    utt2a_probs, utt2a_arcs = read_p1a_chart('{}/p1a.chart'.format(output_dir))
    utt2a_probs_ref, utt2a_arcs_ref = read_p1a_chart('p1a.chart.ref')
    np.testing.assert_allclose(utt2a_probs, utt2a_probs_ref)
    np.testing.assert_allclose(utt2a_arcs, utt2a_arcs_ref)

    print('Checking p1b...')
    os.system('cd {} && bash lab2_p1b.sh > p1b.out'.format(output_dir))
    assert filecmp.cmp('{}/p1b.out'.format(output_dir), 'out/p1b.out')

    print('Checking p2a...')
    os.system('cd {} && bash lab2_p2a.sh > p2a.out'.format(output_dir))
    assert filecmp.cmp('{}/p2a.out'.format(output_dir), 'out/p2a.out')
    gmm = read_gmm('{}/p2a.gmm'.format(output_dir))
    gmm_ref = read_gmm('p2a.gmm.ref')
    np.testing.assert_allclose(gmm, gmm_ref)

    print('Checking p3a...')
    os.system('cd {} && bash lab2_p3a.sh > p3a.out'.format(output_dir))
    assert filecmp.cmp('{}/p3a.out'.format(output_dir), 'out/p3a.out')
    utt2a_forw, utt2a_back, utt2a_post = read_p3a_chart('{}/p3a_chart.dat'.format(output_dir))
    utt2a_forw_ref, utt2a_back_ref, utt2a_post_ref = read_p3a_chart('p3a_chart.ref')
    np.testing.assert_allclose(utt2a_forw, utt2a_forw_ref)
    np.testing.assert_allclose(utt2a_back, utt2a_back_ref)
    np.testing.assert_allclose(utt2a_post, utt2a_post_ref)

    print('Checking p3b...')
    os.system('cd {} && bash lab2_p3b.sh > p3b.out'.format(output_dir))
    assert filecmp.cmp('{}/p3b.out'.format(output_dir), 'out/p3b.out')
    gmm = read_gmm('{}/p3b.gmm'.format(output_dir))
    gmm_ref = read_gmm('p3b.gmm.ref')
    np.testing.assert_allclose(gmm, gmm_ref)
