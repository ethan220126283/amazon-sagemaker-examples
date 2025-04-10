# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new example, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.

## Report Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check [existing open](https://github.com/aws/amazon-sagemaker-examples/issues) and [recently closed](https://github.com/aws/amazon-sagemaker-examples/issues?utf8=%E2%9C%93&q=is%3Aissue%20is%3Aclosed%20) issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps.
* Any modifications you've made relevant to the bug.
* A description of your environment or deployment.

## Contribute via Pull Requests (PRs)

Before sending us a pull request, please ensure that:

* You are working against the latest source on the *main* branch.
* You check the existing open and recently merged pull requests to make sure someone else hasn't already addressed the problem.
* You open an issue to discuss any significant work - we would hate for your time to be wasted.
* **NOTE: If you are submitting an entirely new notebook, please ensure it demonstrates a functionality of SageMaker not yet showcased by any other existing notebook in this repository. If you don't meet this criteria, your PR will be rejected.**

### Pull Down the Code

1. If you do not already have one, create a GitHub account by following the prompts at [Join Github](https://github.com/join).
1. Create a fork of this repository on GitHub. You should end up with a fork at `https://github.com/<username>/amazon-sagemaker-examples`.
   1. Follow the instructions at [Fork a Repo](https://help.github.com/en/articles/fork-a-repo) to fork a GitHub repository.
1. Clone your fork of the repository: `git clone https://github.com/<username>/amazon-sagemaker-examples` where `<username>` is your github username.

### Run the Linter

Apply Python code formatting to Jupyter notebook files using [black](https://pypi.org/project/black/).

1. Install black using `pip3 install black`
1. In terminal, run the following black command on each of your ipynb notebook files and verify that the linter passes: `python3 -m black -l 100 {path}/{notebook-name}.ipynb`
1. Some notebook features such as `%` bash commands or `%%` cell magic cause black to fail. As long as you run the above command to format as much as possible, that is sufficient, even if the check fails

### Test Your Notebook End-to-End

Our [CI system](https://github.com/aws/sagemaker-example-notebooks-testing) runs modified or added notebooks, in parallel, for every Pull Request.
Please ensure that your notebook runs end-to-end so that it passes our CI.

The `sagemaker-bot` will comment on your PR with a link for `Build logs`.
If your PR does not pass CI, you can view the logs to understand how to fix your notebook(s) and code.

### Add CI badges to your notebook

Our [CI system](https://github.com/aws/sagemaker-example-notebooks-testing) tests each notebook in this repo everyday to see if it is fully functional. We provide badges to display the results of these daily tests so you and your customers can see if your notebook is working or needs to be fixed. **It is required that all notebooks have these badges.**

The badges should be added using the following steps:
1. Insert the following markdown underneath your notebook's title. Substitute NOTEBOOK_PATH with the path of your notebook relative to the root of the repo and all "/" are replaced with "|" (i.e. sagemaker-pipeline-parameterization|parameterized-pipeline.ipynb)
```
---

This notebook's CI test result for us-west-2 is as follows. CI test results in other regions can be found at the end of the notebook.

![This us-west-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-west-2/NOTEBOOK_PATH)

---
```

2. Insert the following markdown at the end of your notebook. Substitute all instances of NOTEBOOK_PATH with the path of your notebook relative to the root of the repo and all "/" are replaced with "|" (i.e. sagemaker-pipeline-parameterization|parameterized-pipeline.ipynb)
```
## Notebook CI Test Results

This notebook was tested in multiple regions. The test results are as follows, except for us-west-2 which is shown at the top of the notebook.


![This us-east-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-east-1/NOTEBOOK_PATH)

![This us-east-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-east-2/NOTEBOOK_PATH)

![This us-west-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/us-west-1/NOTEBOOK_PATH)

![This ca-central-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ca-central-1/NOTEBOOK_PATH)

![This sa-east-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/sa-east-1/NOTEBOOK_PATH)

![This eu-west-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-1/NOTEBOOK_PATH)

![This eu-west-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-2/NOTEBOOK_PATH)

![This eu-west-3 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-west-3/NOTEBOOK_PATH)

![This eu-central-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-central-1/NOTEBOOK_PATH)

![This eu-north-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/eu-north-1/NOTEBOOK_PATH)

![This ap-southeast-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-southeast-1/NOTEBOOK_PATH)

![This ap-southeast-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-southeast-2/NOTEBOOK_PATH)

![This ap-northeast-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-northeast-1/NOTEBOOK_PATH)

![This ap-northeast-2 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-northeast-2/NOTEBOOK_PATH)

![This ap-south-1 badge failed to load. Check your device's internet connectivity, otherwise the service is currently unavailable](https://prod.us-west-2.tcx-beacon.docs.aws.dev/sagemaker-nb/ap-south-1/NOTEBOOK_PATH)

```

### Name your notebook

We have migrated to a new standarized naming convention for all notebooks with the repository. The naming format follows the pattern: sm - {name_of_sagemaker_feature} _ {any_key_secondary_feature} _ {detailed_description_of_notebook_focus} . ipynb

Examples:
- sm-jumpstart_foundation_trainium_inferentia_finetuning_deployment.ipynb
- sm-training_compiler_language_modeling_multi_gpu_multi_node.ipynb
- sm-clarify_text_explainability_text_sentiment_analysis.ipynb

###  Place your notebook into the correct folder

We have implemented a flattened directory structure in order to increase the discoverability of notebooks within the repository. Once you have completed the notebook, place it into the folder that best corresponds with the primary functionality that you highlighting within your example notebook. Here is a list of the folders and a brief description of their primary purposes:

- end_to_end_ml_lifecycle - end-to-end notebooks that demonstrate how to build, train, and deploy machine learning models using Amazon SageMaker
- prepare_data - noteboooks that showcase Amazon SageMaker's data preparation capabilities
- building_and_train_models - notebooks that highlight Amazon SageMaker tools to build and train ML models at scale
- deploy_and_monitor - notebooks that demonstrate Amazon SageMaker's ML infrastructure and model deployment options as well as SageMaker's ability to monitor the quality of your machine learning models in real time
- responsible_ai - notebooks that highlight Amazon SageMaker's abilities to improve your machine learning models by detecting potential bias and helping to explain the predictions that your models make from your tabular, computer vision, natural processing, or time series datasets
- ml_ops - notebooks that feature Amazon SageMaker's ability to implement machine learning models in production environments with continuous integration and deployment
- generative_ai - notebooks that demonstate Amazon SageMaker's generative AI capabilities to create new, synthetic data across various modalities, such as text, images, audio, and video, based on the patterns and relationships learned from training data

### Add Your Notebook to the Website

#### Environment Setup

You can use the same Conda environment for multiple related projects. This means you can add a few dependencies and update the environment as needed.

1. You can do this by using an environment file to update the environment
2. Or just use conda or pip to install the new deps
3. Update the name (the -n arg) to whatever makes sense for you for your project
4. Keep an eye out for updates to the dependencies. This project’s dependencies are here: https://github.com/aws/amazon-sagemaker-examples/blob/main/environment.yml
5. Fork the repo: https://github.com/aws/amazon-sagemaker-examples.git
6. Clone your fork
7. Cd into the fork directory
8. Create and activate your environment. You can likely use a higher version of Python, but RTD is currently building with 3.6 on production

```
# Create the env
conda create -n sagemaker-examples python=3.6
# Activate it
conda activate sagemaker-examples
```

Install dependencies:

```
# Install deps from environment file
conda env update -f environment.yml
```

#### Dependency Notes:

When you build, there’s a bunch of warnings about a python3 lexer not found. Solution is here: https://github.com/spatialaudio/nbsphinx/issues/24
Although this workaround required add the following to conf.py and pinning prompt-toolkit as it requires a downgrade to work with the IPython package coming from conda.
`"IPython.sphinxext.ipython_console_highlighting"`

**Follow-up for next round of dependency updates:** Another workaround could be to use the pip IPython package instead of the conda one (there’s mention the conda one might be buggy), then maybe you don’t need to add that to conf.py or fix prompt-toolkit.

#### Adjusting navigation

Some pages have nested title elements that will impact the navigation and depth. The following shows the title, using the top and bottom hash marks (####). Then the single line equals sign (====), then the dashes (----). These are equivalent to H1, H2, and H3, respectively.

```
################
AWS Marketplace
################

Publish algorithm on the AWS Marketplace
===========================================

Create your algorithm and model package
----------------------------------------

.. toctree::
   :maxdepth: 1

   creating_marketplace_products/algorithms/Bring_Your_Own-Creating_Algorithm_and_Model_Package
```

You can create further depth by using tilde (~~~~~), asterisk (********), and caret (^^^^^).

Important: the underline must be at least as long as the title you’re underlining.

#### Adjusting content display

Typically you want to use  `:maxdepth: 1`

You can adjust how much detail from a notebook appears on a page by changing `maxdepth`. Zero and one depth are the same, and these will display just the title. This would be the H1 element for the notebook. Setting this to 2 would display the H2 elements (## Some subtitle) as well.

Sometimes you include topics from other folders on one index page. If you include a subfolder’s index in the TOC using maxdepth of 1, you might just get one entry. So this is an instance where updating maxdepth to 2 would yield a better result.

If more than one entry is displayed for the same notebook, this is because the author of the notebook mistakenly used multiple H1’s. You can see this in the notebooks where they do this:

```
# Main title [CORRECT]
Some content

## Subtitle
Some content

# Some other section [INCORRECT]
Some content
```

Then you’ll get a two bullets (the extra “Some other section” when there should only be one for the main title.

#### Troubleshooting

* Each notebook should have at least one section title

> /Users/markhama/Development/amazon-sagemaker-examples/r_examples/r_batch_transform/r_xgboost_batch_transform.ipynb:6: WARNING: Each notebook should have at least one section title

This means the author doesn’t have a title in the notebook. The first markdown block should have a title like `# Some fancy title`. In some cases the author used html tags like `<h1>`. These render fine on GitHub, but will error in the website build causing the notebook to be skipped.

* toctree contains reference to nonexisting document

> ~/Development/amazon-sagemaker-examples/r_examples/index.rst:5: WARNING: toctree contains reference to nonexisting document 'r_examples/r_batch_transform/r_xgboost_batch_tranform'

Check your spelling in the notebook’s path.


* Notebook has an entry, but the title seems incorrect.

Check the notebook for the title (# Some title). The author likely didn’t conform to title/subtitle hierarchy in markdown.


### Commit Your Change

Use imperative style and keep things concise but informative. See [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) for guidance.

### Send a Pull Request

GitHub provides additional document on [Creating a Pull Request](https://help.github.com/articles/creating-a-pull-request/).


Please remember to:
* Send us a pull request, answering any default questions in the pull request interface.
* Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.


## Writing Sequential Notebooks

Most notebooks are singular - only one notebook (.ipynb file) is needed to run that example. However, there are a few cases in which an example may be split into multiple notebooks. These are called sequential notebooks, as the sequence of the example is split among multiple notebooks. An example you can look at is [this series of sequential notebooks that demonstrate how to build a music recommender](https://github.com/aws/amazon-sagemaker-examples/tree/main/end_to_end/music_recommendation).

### When should Sequential Notebooks be used?

You may want to consider using sequential notebooks to write your example if the following conditions apply:

* Your example takes over two hours to execute.
* You want to emphasize on the different steps of the example in great detail and depth (i.e. one notebook goes into detail about data exploration, the next notebook thoroughly describes the model training process, etc).
* You want customers to have the ability to run part of your example if they wish to (i.e. they only want to run the training portion).

### What are the guidelines for writing Sequential Notebooks?

If you determine that sequential notebooks are the most suitable format to write your examples, please follow these guidelines:

* *Each notebook in the series must independently run end-to-end so that it can be tested in the daily CI (i.e. the CI test amazon-sagemaker-example-pr must pass).*
    * This may include generating intermediate artifacts which can be immediately loaded up for use in later notebooks, etc. Depending on the situation, intermediate artifacts can be stored in the following places: 
        * The repo in the same folder where your notebook is stored: This is possible for very small files (on the order of KB)
        * The sagemaker-example-files-prod-REGION S3 bucket: This is for larger files (on or above the order of MB).
* Each notebook must have a 'Background Section' clearly stating that the notebook is part of a notebook sequence. It must contain the following elements below. You can look at the 'Background' section in [Music Recommender Data Exploration](https://github.com/aws/amazon-sagemaker-examples/blob/main/end_to_end/music_recommendation/01_data_exploration.ipynb) for an example.
    * The objective and/or short summary of the notebook series.
    * A statement that the notebook is part of a notebook series.
    * A statement communicating that the customer can choose to run the notebook by itself or as part of the series.
    * List and link to the other notebooks in the series.
    * Clearly display where the current notebook fits in relation to the other notebooks (i.e. it is the 3rd notebook in the series).
    * If you have a README that contains more introductory information about the notebook series as a whole, link to it. For example, it is nice to have an architecture diagram showing how the services interact across different notebooks - the README would be a good place to put such information. An example of such a README is You can look at this [README.md](https://github.com/aws/amazon-sagemaker-examples/blob/main/end_to_end/music_recommendation/README.md).
* If you have a lot of introductory material for your series, please put it in a README that is located in the same directory with your notebook series instead of an introductory notebook. You can look at this [README.md](https://github.com/aws/amazon-sagemaker-examples/blob/main/end_to_end/music_recommendation/README.md) as an example.
* When you first use an intermediate artifact in a notebook, add a link to the notebook that is responsible for generating that artifact. That way, customers can easily look up how that artifact was created if they wanted to.
* Use links to shorten the length of your notebook and keep it simple and organized. Instead of writing a long passage about how a feature works (i.e Batch Transform), it is better to link to the documentation for it. 
* Design your notebook series such that the customer can get benefit from both the individual notebooks and the whole series. For example, each notebook should have clear takeaway points for the customer (i.e. one notebook teaches data preparation and feature engineering, the next notebook teaches training, etc).
* Put the sequence order in the notebook file name. For example, the first notebook should start with "1_", the second notebook with "2_", etc.


## Example Notebook Best Practices

Here are some general guidelines to follow when writing example notebooks:
* Use the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) wherever possible, rather than `boto3`.
* Do not hardcode information like security groups, subnets, regions, etc.
    ```python
    # Good
    loader = botocore.loaders.create_loader()
    resolver = botocore.regions.EndpointResolver(loader.load_data("endpoints"))
    resolver.construct_endpoint("s3", region)

    # Bad
    cn_regions = ['cn-north-1', 'cn-northwest-1']
    region = boto3.Session().region_name
    endpoint_domain = 'com.cn' if region in cn_regions else 'com'
    's3.{}.amazonaws.{}'.format(region, endpoint_domain)
    ```
* Do not require user input to run the notebook.
  * 👍 `bucket = session.default_bucket()`
  * 👎 `bucket = <YOUR_BUCKET_NAME_HERE>`
* Lint your code and notebooks. (See the section on [running the linters](#run-the-linters) for guidance.)
* Use present tense.
  * 👍 "The estimator fits a model."
  * 👎 "The estimator will fit a model."
* When referring to an AWS product, use its full name in the first invocation.
  (This applies only to prose; use what makes sense when it comes to writing code, etc.)
  * 👍 "Amazon S3"
  * 👎 "s3"
* Provide links to other ReadTheDocs pages, AWS documentation, etc. when helpful.
  Try to not duplicate documentation when you can reference it instead.
  * Use meaningful text in a link.
    * 👍 You can learn more about [hyperparameter tuning with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html) in the SageMaker docs.
    * 👎 Read more about it [here](#).


## Find Contributions to Work On

Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels ((enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any ['help wanted'](https://github.com/aws/amazon-sagemaker-examples/labels/help%20wanted) issues is a great place to start.


## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security Issue Notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](https://github.com/aws/amazon-sagemaker-examples/blob/main/LICENSE.txt) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
