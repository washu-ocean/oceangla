What is oceangla?
=================

``oceangla`` is a program that performs group-level analyses on task-based fMRI data. It operates
on the outputs of ``oceanfla``, the OCEAN's first-level analysis package.

``oceangla`` aims to make this process simple by providing an R-style formula syntax which can
a) wrangle/organize the appropriate effect maps and behavioral data into memory and b) deduce how
a multiple linear regression should work based off of this input.

Quickstart: a simple OLS model
------------------------------

.. highlight:: bash
These models can be specified on the command line, as in this example where we're getting the main effect of 
age in some arbitrary task effect "taskresponse"::

  oceangla \
    -f /path/to/oceanfla_outputs \
    -o /path/to/oceangla_outputs \
    --csv /path/to/subject_specific_variables.csv \
    --model-name main_effect_age \
    --model 'taskresponse ~ age'

Breaking this down:

* ``-f /path/to/oceanfla_outputs``: This is where ``oceangla`` will search for first-level outputs for
  each subject. You can specify multiple FLA directories to pull from by separating them with spaces
  after the initial ``-f`` for example::

    -f /path/to/oceanfla_outputs1 /path/to/oceanfla_outputs2 ... /path/to/oceanfla_outputsN

* ``-o /path/to/oceangla_outputs``: This is where ``oceangla`` will search for first-level outputs for
  each subject.

* ``--csv /path/to/subject_specific_variables.csv``: This is where ``oceangla`` will pull any behavioral (or other subject-specific) measures.
  The right side of the model formula after the ``~`` should contain variables matching column names in this .csv file. .tsv files can also be
  used with the ``--tsv`` flag. Multiple .csv or .tsv files can also used by separating them with spaces, as with the ``-f`` flag.

* ``--model-name main_effect_age``: This is the name the following model can be identified by.
  In this example, the outputs of this model will be stored in a folder named "main_effect_age" under the
  directory specifyied by ``-o``, like so::

    /path/to/oceangla_outputs/main_effect_age

* ``--model 'taskresponse ~ age'``: What appears in the single quotes after ``--model`` is the formula
  defining this group-level GLM: in this case, we want to understand the relationship between the variable ``age`` 
  for each subject and the ``taskresponse`` effect size at each voxel for all subjects. ``oceangla`` will 
  find all first-level statistical maps matching the ``taskresponse`` effect, pair each map with the corresponding
  ``age`` variable for each subject, then run an OLS model that will return t-statistics and p-values for an
  intercept term and ``age``. 

If you want to run multiple models, you'll need to specify both the ``--model-name`` and ``--model`` for each formula
you want to run; for example::

  oceangla \
    -f /path/to/oceanfla_outputs \
    -o /path/to/oceangla_outputs \
    --csv /path/to/subject_specific_variables.csv \
    --model-name main_effect_age \
    --model 'taskresponse ~ age'
    --model-name main_effect_adhd \
    --model 'taskresponse ~ adhd_score'

Each ``--model-name`` will be associated with each ``--model`` formula in the order they are specified -- so above, 
'main_effect_age' will be associated with the 'taskresponse ~ age' model, and 'main_effect_adhd' will be associated
with 'taskresponse ~ adhd_score'. 

.. warning::

  Make sure each formula specified with ``--model`` is wrapped in single quotes ``'``!
  Without quotes, oceangla will not parse the formula correctly (and probably crash).


.. warning::

    When specifying your model formula, any effect names that appear hyphenated in their filename 
    under the FLA folder -- like "One-Player" in the following example::

      sub-1000_ses-1_task-SomeTask_space-MNI152_condition-One-Player_stat-effect_boldmap.nii.gz

    should have their hyphens replaced with underscores '_'. This is becasue the hyphen '-'
    character is used as a subtraction symbol in this context, so that contrasts between first-level effects
    can be generated on the fly (more on that later!).

    Bad formula example::

        oceangla ... --model 'One-Player ~ age' # will try to create a contrast between conditions 'One' and 'Player'

    
    Good formula example::

        oceangla ... --model 'One_Player ~ age' # works!


.. hightlight:: none

Alternate methods for specifying multiple models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a couple of ways to shorten the commands above, which is especially helpful if you want to run multiple models
at a time or are iterating on lots of different variables. The current alternate methods are:

1. Instead of specifying a ``--model-name`` and ``--model`` for each model, you can specify a ``--model-file`` that
   contains every model you'd like to run. This should be a simple text file where there is one model name and formula
   separated by a ``->`` on each line. For example, you can create a file ``models.txt`` with the following contents::

    main_effect_age -> taskresponse ~ age
    main_effect_adhd -> taskresponse ~ adhd
    interaction_age_adhd -> taskresponse ~ age * adhd

.. highlight:: bash

then run a similar command as above using ``--model-file`` instead::
  
  oceangla \
    -f /path/to/oceanfla_outputs \
    -o /path/to/oceangla_outputs \
    --csv /path/to/subject_specific_variables.csv \
    --model-file /path/to/models.txt
  
2. To take a "kitchen sink" approach and run each individual first-level variable through the same model (for example,
   model every first-level variable as a main effect of age), you can run it with the following command::

    oceangla \
      -f /path/to/oceanfla_outputs \
      -o /path/to/oceangla_outputs \
      --csv /path/to/subject_specific_variables.csv \
      --model-name main_effect_age \
      --model 'ALL ~ main_effect_age'


.. warning::

  Using the 'ALL' shorthand above will run your model on *all* first-level effects, including any nuisance regressors.




