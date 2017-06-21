#!/usr/bin/python3
import argparse
import os
import sys

from processors import *


def is_valid_directory(dir, text, create=False):
    if create and not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.isdir(dir):
        print('Error: {} ({})'.format(text, dir))
        sys.exit(1)


def is_valid_file(file, text):
    if not os.path.isfile(file):
        print('Error: {} ({})'.format(text, file))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Replaces the green background for a series of images with a background image in two steps', \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--task', choices=['both', 'pre', 'post'], required=True, default='both', \
                        help='select whether preprocessing or postprocessing will be done')
    parser.add_argument('-s', '--sourcedir', default='test_images', \
                        help='directory where imput images are stored')
    parser.add_argument('--datafile', default='/tmp/greenkey_bg/datafile.csv', \
                        help='file where the tool stores its preprocessing results resp. postprocessing results are loaded from.')

    parser_pre = parser.add_argument_group('PreProcessing', 'Options which only affects preprocessing.')
    parser_pre.add_argument('-f', '--filter', default='zoom14.jpg', \
                            help='file endings matching the filter value will be processed')
    parser_pre.add_argument('--tmpdir', default='/tmp/greenkey_bg', \
                            help='location of the preprocessing output. This might be usefull for separate postprocessing')

    parser_post = parser.add_argument_group('PostProcessing', 'Options which only affects postprocessing.')
    parser_post.add_argument('-o', '--outdir', default='out', \
                             help='directory where output images are placed')
    parser_post.add_argument('-b', '--bgimg', default='test_images/background.jpg', \
                             help='image used as background')
    parser_post.add_argument('-m', '--usemin', default=False, action='store_true', help='uses the minimum width of all object as target')

    args = parser.parse_args()

    is_valid_directory(args.sourcedir, 'sourcedir does not exist!')
    is_valid_directory(args.tmpdir, 'tmpdir does not exist!', create=True)
    is_valid_file(args.bgimg, 'background image does not exist!')


    if args.task in ['pre', 'both']:
        print('Start preprocessing...')

        result = PreProcessingResults()

        for in_file in os.listdir(args.sourcedir):
            if in_file.endswith(args.filter):
                p = PreProcessor(result, args.sourcedir, args.tmpdir)
                p.process_file(in_file)

        # print and safe results
        result.print()
        head, tail = os.path.split(args.datafile)
        if not os.path.exists(head):
            args.datafile = args.tmpdir + "/" + tail
        result.save(args.datafile)
        print('preprocessing done')
        args.sourcedir = args.tmpdir

    if args.task in ['post', 'both']:
        print('Start postprocessing...')

        # load preprocessing results
        is_valid_file(args.datafile, 'Preprocessing data file does not exist!')
        result = PreProcessingResults()
        result.load(args.datafile)

        # initialize and calculate destination size
        post = PostProcessor(result, args.bgimg, args.usemin)

        is_valid_directory(args.outdir, 'outdir does not exist!', create=True)

        for r in result._result:
            out_filepath = os.path.join(args.outdir, r.filename_in)
            post.addBackground(r, out_filepath)

        print('preprocessing done')


if __name__ == "__main__":
    main()
