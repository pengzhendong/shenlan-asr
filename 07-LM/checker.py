import os
import sys
import filecmp


if __name__ == '__main__':
    output_dir = sys.argv[1]
    os.system('make clean -C {}'.format(output_dir))
    os.system('make -C {}'.format(output_dir))

    print('Checking p1a...')
    os.system('cd {} && bash lab3_p1a.sh > p1a.out'.format(output_dir))
    assert filecmp.cmp('{}/p1a.out'.format(output_dir), 'out/p1a.out')

    print('Checking p1b...')
    os.system('cd {} && bash lab3_p1b.sh > p1b.out'.format(output_dir))
    assert filecmp.cmp('{}/p1b.out'.format(output_dir), 'out/p1b.out')

    print('Checking p3a...')
    os.system('cd {} && bash lab3_p3a.sh > p3a.out'.format(output_dir))
    assert filecmp.cmp('{}/p3a.out'.format(output_dir), 'out/p3a.out')

    print('Checking p3b...')
    os.system('cd {} && bash lab3_p3b.sh > p3b.out'.format(output_dir))
    assert filecmp.cmp('{}/p3b.out'.format(output_dir), 'out/p3b.out')
