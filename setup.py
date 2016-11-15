#!/usr/bin/env python

### setup.py

#from distutils.core import setup

from setuptools import setup, find_packages
import sys, os

setup(name='DecisionTree',
      version='3.4.3',
      author='Avinash Kak',
      author_email='kak@purdue.edu',
      maintainer='Avinash Kak',
      maintainer_email='kak@purdue.edu',
      url='https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html',
      download_url='https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.tar.gz#md5=cf0e589093499aeda9ec94f057685d6b',
      description='A Python module for decision-tree based classification of multidimensional data',
      long_description=''' 



Consult the module API page at 

      https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html

for all information related to this module, including
information regarding the latest changes to the code. The
page at the URL shown above lists all of the module
functionality you can invoke in your own code.  That page
also describes in great detail how you can use the boosting
and the bagging capabilities of the module, and the
capabilities allowed by the new RandomizedTreesForBigData
class that was introduced in Version 3.3.0.  Recent changes
to the module allow you to tackle needle-in-a-haystack and
big-data classification problems.  The needle-in-a-haystack
metaphor is useful when your training data is excessively
dominated by just one class.

With regard to the basic purpose of the module, assuming you
have placed your training data in a CSV file, all you have
to do is to supply the name of the file to this module and
it does the rest for you without much effort on your part
for classifying a new data sample.  A decision tree
classifier consists of feature tests that are arranged in
the form of a tree. The feature test associated with the
root node is one that can be expected to maximally
disambiguate the different possible class labels for a new
data record.  From the root node hangs a child node for each
possible outcome of the feature test at the root. This
maximal class-label disambiguation rule is applied at the
child nodes recursively until you reach the leaf nodes.  A
leaf node may correspond either to the maximum depth desired
for the decision tree or to the case when there is nothing
further to gain by a feature test at the node.

Typical usage syntax:

::

      training_datafile = "stage3cancer.csv"
      dt = DecisionTree.DecisionTree( 
                      training_datafile = training_datafile,
                      csv_class_column_index = 2,
                      csv_columns_for_features = [3,4,5,6,7,8],
                      entropy_threshold = 0.01,
                      max_depth_desired = 8,
                      symbolic_to_numeric_cardinality_threshold = 10,
           )

        dt.get_training_data()
        dt.calculate_first_order_probabilities()
        dt.calculate_class_priors()
        dt.show_training_data()
        root_node = dt.construct_decision_tree_classifier()
        root_node.display_decision_tree("   ")

        test_sample  = ['g2 = 4.2',
                        'grade = 2.3',
                        'gleason = 4',
                        'eet = 1.7',
                        'age = 55.0',
                        'ploidy = diploid']
        classification = dt.classify(root_node, test_sample)
        print "Classification: ", classification

          ''',

      license='Python Software Foundation License',
      keywords='data classification, decision trees, data analytics, regression',
      platforms='All platforms',
      classifiers=['Topic :: Scientific/Engineering :: Information Analysis', 'Programming Language :: Python :: 2.7', 'Programming Language :: Python :: 3.4'],
      packages=['DecisionTree','DecisionTreeWithBagging','BoostedDecisionTree','RandomizedTreesForBigData','RegressionTree']
)
