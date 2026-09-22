---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Just checking in on the infrastructure. I'm sure it's nothing, but the database
    cluster has stopped replicating and data integrity is currently failing across
    all shards. Please let me know if you need any tea or coffee while you handle
    this.
- text: NAS Volume Inaccessible. Our departmental NAS (QNAP TS-873A) is showing a
    'Volume Degraded' status. Users report they cannot map the network drive to their
    Windows 10 workstations. I've tried restarting the service, but it didn't help.
- text: 'Elevated privileges required for software install. I need to install Python
    3.12 for a dev project, but my standard user account is restricted. Please grant
    temporary admin rights to my machine (Asset ID: LAP-9982) so I can finish this
    by EOD.'
- text: Server room cooling alarm. The HVAC sensor in the server closet is reading
    88 degrees Fahrenheit. I don't hear the blower running. Please fix this before
    my servers melt.
- text: Application timeout during peak hours. Users are reporting 504 Gateway Timeout
    errors when attempting to save records in the CRM during mid-morning. It is unclear
    if the SQL query is hanging or if the load balancer is dropping packets.
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/all-MiniLM-L6-v2
---

# SetFit with sentence-transformers/all-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 7 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label             | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|:------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Access Management | <ul><li>"Need access to the Marketing Share Drive. Hi, I just moved over from the Sales department. Can you please add me to the M:\\Marketing folder? I can currently see the drive but get an 'Access Denied' error when trying to open it."</li><li>"Cannot access shared HR drive. I am trying to open the Z: drive but I keep getting an 'Access Denied' error. My manager said I should have permissions for the 'HR_Restricted' folder. I am running Windows 11 Pro."</li><li>'SSH access request to dev-srv-04. I need SSH access to dev-srv-04 for the upcoming performance testing. My public key is attached. Could you add me to the authorized_keys file?'</li></ul>                                                                                                                                                                                                                                                                                                                                                                |
| Network           | <ul><li>'VPN login timeout. Trying to log in and it just hangs at 40%.'</li><li>'Packet loss on guest network. Multiple visitors have reported that they cannot load basic webpages while connected to the Guest_WiFi network. Signal strength is excellent, but DNS resolution seems to be failing intermittently.'</li><li>"Print jobs stuck in spooler. The ERP system sends print jobs that never arrive at the network printer. The spooler shows 'Error' status."</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Storage           | <ul><li>"NAS volume unmounted unexpectedly. Our QNAP TS-h886 storage server suddenly dropped the iSCSI target connection around 02:00 AM. Several VMs hosted on this volume are currently in a paused state. We are seeing 'Target portal unreachable' errors in the vSphere logs."</li><li>"USB External drive not detected. I plugged my 2TB Western Digital drive into the USB 3.0 port on my workstation and nothing happens. The light on the drive blinks, but it doesn't show up in Disk Management. Frustrating because I have a deadline today."</li><li>"Local storage full on design workstation. My drive is full again! I keep getting low disk space warnings on my MacBook Pro. I already deleted my downloads folder, but I can't finish my current project because the system won't let me save."</li></ul>                                                                                                                                                                                                                     |
| Infrastructure    | <ul><li>"Can't get into the server room. The HID badge reader on the primary server room door is completely dead. I've swiped three different badges and nothing happens. The light on the reader isn't even turning on."</li><li>'UPS Battery Warning. The APC Smart-UPS 3000 in the server room is throwing a "Battery Replacement Required" error on the LCD again. The input voltage is perfectly stable at 120V, yet the unit is incessantly beeping and driving everyone crazy. I need this dealt with immediately.'</li><li>'Rack A-02 Power Outage. The entire rack in A-02 is completely dead and the PDU is showing a 0A load. This is ridiculous—I’m almost certain the breaker in the facility room tripped. Fix this immediately.'</li></ul>                                                                                                                                                                                                                                                                                        |
| Security          | <ul><li>"Encryption error on external drive. I'm trying to backup some sensitive data to an encrypted WD My Passport drive, but the encryption software fails with Error Code 0x80040154. I've tried different ports on my workstation, but it's the same error."</li><li>"System slow after suspected malware. My computer has been incredibly slow since I accidentally clicked a link earlier. CPU usage for 'System' is pegged at 100%. I'm worried it's a crypto-miner."</li><li>'User account locked out. I am completely locked out of my PC again. I’ve only tried logging in twice with the correct password, and now it’s giving me an "Account Locked" error. This is incredibly frustrating and a total waste of my time. Please fix this immediately so I can actually get some work done.'</li></ul>                                                                                                                                                                                                                               |
| Application       | <ul><li>'Adobe Creative Cloud login loop. Every time I try to launch Photoshop on my Windows 11 workstation, it prompts me to sign in, accepts my credentials, and then immediately boots me back to the sign-in screen. I have already cleared the OOBE and SLStore folders as suggested on the forums, but the issue persists. My machine ID is LPT-8842.'</li><li>"Adobe Acrobat crashes on startup. I am beyond fed up with Adobe Acrobat Pro DC. Every single time I try to open a PDF, the application crashes immediately. I’ve already wasted my time doing a full reinstallation of the entire Creative Cloud suite, and it didn't do a damn thing. I'm running Windows 11 Pro 22H2, and the Event Viewer is clearly pointing to the faulting module name: Acrobat.dll, version: 23.8.20470.0. Get this fixed, I have actual work to do."</li><li>"Zoom audio echo. Hi, people on my team calls keep complaining that they hear themselves echoing when I speak. I'm using a Jabra headset on my iMac M2. How do I fix this?"</li></ul> |
| Database          | <ul><li>'MongoDB primary node flapping. The primary node in our MongoDB 5.0 replica set is restarting intermittently. Error logs indicate a heartbeat timeout. This is causing significant application errors every few minutes.'</li><li>'Performance degradation after migration. Ever since we migrated from SQL Server 2016 to 2022, the report generation service is incredibly slow. We are getting timeouts on the report_gen_sp procedure. Everything worked fine yesterday before the upgrade.'</li><li>"Unable to export data to CSV. Every time I try to run the export tool on the web interface, it just spins for a minute and then returns a 500 error. The logs say 'Memory limit exceeded'."</li></ul>                                                                                                                                                                                                                                                                                                                          |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Server room cooling alarm. The HVAC sensor in the server closet is reading 88 degrees Fahrenheit. I don't hear the blower running. Please fix this before my servers melt.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 6   | 38.7612 | 100 |

| Label             | Training Sample Count |
|:------------------|:----------------------|
| Access Management | 97                    |
| Application       | 99                    |
| Database          | 96                    |
| Infrastructure    | 101                   |
| Network           | 110                   |
| Security          | 99                    |
| Storage           | 93                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 20
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: True

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0006 | 1    | 0.4241        | -               |
| 0.0288 | 50   | 0.2867        | -               |
| 0.0575 | 100  | 0.2291        | -               |
| 0.0863 | 150  | 0.2055        | -               |
| 0.1151 | 200  | 0.1630        | -               |
| 0.1438 | 250  | 0.1449        | -               |
| 0.1726 | 300  | 0.1398        | -               |
| 0.2014 | 350  | 0.1132        | -               |
| 0.2301 | 400  | 0.0948        | -               |
| 0.2589 | 450  | 0.0863        | -               |
| 0.2877 | 500  | 0.0827        | -               |
| 0.3165 | 550  | 0.0665        | -               |
| 0.3452 | 600  | 0.0747        | -               |
| 0.3740 | 650  | 0.0522        | -               |
| 0.4028 | 700  | 0.0525        | -               |
| 0.4315 | 750  | 0.0527        | -               |
| 0.4603 | 800  | 0.0548        | -               |
| 0.4891 | 850  | 0.0449        | -               |
| 0.5178 | 900  | 0.0501        | -               |
| 0.5466 | 950  | 0.0415        | -               |
| 0.5754 | 1000 | 0.0448        | -               |
| 0.6041 | 1050 | 0.0460        | -               |
| 0.6329 | 1100 | 0.0460        | -               |
| 0.6617 | 1150 | 0.0301        | -               |
| 0.6904 | 1200 | 0.0359        | -               |
| 0.7192 | 1250 | 0.0296        | -               |
| 0.7480 | 1300 | 0.0420        | -               |
| 0.7768 | 1350 | 0.0280        | -               |
| 0.8055 | 1400 | 0.0330        | -               |
| 0.8343 | 1450 | 0.0234        | -               |
| 0.8631 | 1500 | 0.0259        | -               |
| 0.8918 | 1550 | 0.0276        | -               |
| 0.9206 | 1600 | 0.0267        | -               |
| 0.9494 | 1650 | 0.0274        | -               |
| 0.9781 | 1700 | 0.0280        | -               |
| 1.0    | 1738 | -             | 0.1319          |

### Framework Versions
- Python: 3.11.9
- SetFit: 1.2.0
- Sentence Transformers: 5.6.1
- Transformers: 5.14.1
- PyTorch: 2.13.0+cpu
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->