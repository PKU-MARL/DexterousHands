
==========================
Environments
==========================

Source code for tasks can be found in `dexteroushandenvs/tasks`. 

Until now we only suppose the following environments:

+--------------+-----------------+-------------------------+-------------------------+----------------------------+-------------------------+
| Environments | ShadowHandOver  | ShadowHandCatchUnderarm | ShadowHandCatchOverarm  | ShadowHandTwoCatchUnderarm | ShadowHandCatchAbreast  |
+==============+=================+=========================+=========================+============================+=========================+
| Actions Type | Continuous      | Continuous              | Continuous              | Continuous                 | Continuous              |
+--------------+-----------------+-------------------------+-------------------------+----------------------------+-------------------------+
| Agents Num   | 2               |  2                      | 2                       | 2                          | 2                       |
+--------------+-----------------+-------------------------+-------------------------+----------------------------+-------------------------+
|Action Shape  |(num_envs, 2, 20)|(num_envs, 2, 26)        |(num_envs, 2, 26)        | (num_envs, 2, 26)          | (num_envs, 2, 26)       |
+--------------+-----------------+-------------------------+-------------------------+----------------------------+-------------------------+
|Action Value  |[-1, 1]          |[-1, 1]                  |[-1, 1]                  | [-1, 1]                    | [-1, 1]                 |
+--------------+-----------------+-------------------------+-------------------------+----------------------------+-------------------------+

.. important:: 
  Still under development


ShadowHandOver Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../assets/image_folder/0.gif

ShadowHandCatchUnderarm Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../assets/image_folder/1.gif


ShadowHandCatchAbreast Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../assets/image_folder/5.gif