# Dataset 1 - KDD Cup 1999 Data

link: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html



Table 1: Basic features of individual TCP connections
| Feature name   | Description                                                            | Type       |
|----------------|------------------------------------------------------------------------|-------------|
| duration       | length (number of seconds) of the connection                            | continuous  |
| protocol_type  | type of the protocol, e.g. tcp, udp, etc.                               | discrete    |
| service        | network service on the destination, e.g., http, telnet, etc.            | discrete    |
| src_bytes      | number of data bytes from source to destination                         | continuous  |
| dst_bytes      | number of data bytes from destination to source                         | continuous  |
| flag           | normal or error status of the connection                                | discrete    |
| land           | 1 if connection is from/to the same host/port; 0 otherwise              | discrete    |
| wrong_fragment | number of “wrong” fragments                                             | continuous  |
| urgent         | number of urgent packets                                                | continuous  |

Table 2: Content features within a connection suggested by domain knowledge
| Feature name       | Description                                                     | Type       |
|--------------------|-----------------------------------------------------------------|-------------|
| hot                | number of “hot” indicators                                      | continuous  |
| num_failed_logins  | number of failed login attempts                                 | continuous  |
| logged_in          | 1 if successfully logged in; 0 otherwise                        | discrete    |
| num_compromised    | number of “compromised” conditions                              | continuous  |
| root_shell         | 1 if root shell is obtained; 0 otherwise                        | discrete    |
| su_attempted       | 1 if “su root” command attempted; 0 otherwise                   | discrete    |
| num_root           | number of “root” accesses                                       | continuous  |
| num_file_creations | number of file creation operations                              | continuous  |
| num_shells         | number of shell prompts                                         | continuous  |
| num_access_files   | number of operations on access control files                    | continuous  |
| num_outbound_cmds  | number of outbound commands in an ftp session                   | continuous  |
| is_hot_login       | 1 if the login belongs to the “hot” list; 0 otherwise           | discrete    |
| is_guest_login     | 1 if the login is a “guest” login; 0 otherwise                  | discrete    |

Table 3: Traffic features computed using a two-second time window
| Feature name      | Description                                                                                         | Type       |
|-------------------|-----------------------------------------------------------------------------------------------------|-------------|
| count             | number of connections to the same host as the current connection in the past two seconds             | continuous  |
| serror_rate       | % of connections that have “SYN” errors                                                             | continuous  |
| rerror_rate       | % of connections that have “REJ” errors                                                             | continuous  |
| same_srv_rate     | % of connections to the same service                                                                | continuous  |
| diff_srv_rate     | % of connections to different services                                                              | continuous  |
| srv_count         | number of connections to the same service as the current connection in the past two seconds          | continuous  |
| srv_serror_rate   | % of connections that have “SYN” errors                                                             | continuous  |
| srv_rerror_rate   | % of connections that have “REJ” errors                                                             | continuous  |
| srv_diff_host_rate| % of connections to different hosts                                                                 | continuous  |


# Dataset 2 - Network data schema in the Netflow V9 format

link: https://www.kaggle.com/datasets/ashtcoder/network-data-schema-in-the-netflow-v9-format

The dataset contains 44 network features and includes both normal traffic and three main types of intrusion attacks that were executed in a controlled environment. The researchers developed machine learning models to detect these attacks with high accuracy, achieving nearly perfect performance with Random Forest and Gradient Boosted Trees algorithms.

Network Scanning (Reconnaissance)
SYN Scan attacks represent the reconnaissance phase where attackers probe network infrastructure. This type relies on the TCP three-way handshake mechanism, where attackers send SYN packets to desired ports and analyze responses to determine port states (open, closed, or filtered). The dataset contains 2,496,814 flows of SYN Scan attack traffic. Tools like nmap and Masscan were used to generate this attack traffic by scanning IP networks within the target infrastructure.​

Denial-of-Service Attacks
The dataset includes two sophisticated DoS attack variants that operate through slow, legitimate-appearing HTTP traffic rather than traditional flooding:

Denial-of-Service using SlowLoris comprises 864,054 flows in the dataset. This attack opens numerous HTTP connections to targets and sends incomplete but legitimate HTTP requests at an extremely slow pace, keeping connections alive for extended periods. The traffic appears legitimate, mimicking slow clients, which makes detection challenging through traditional methods. By occupying available connections, the attack causes targets to respond slowly to legitimate users or become completely unresponsive.​

Denial-of-Service using R-U-Dead-Yet (RUDY) contains 2,276,947 flows. While similar to SlowLoris in its slow-paced approach, RUDY specifically targets application processing power by sending many small HTTP POST messages (typically 1 byte of data) rather than HTTP headers. This occupies processing resources and maintains numerous open connections, gradually degrading service availability.

# Dataset 3 - cores-iot

We don't have any details in provided dataset, just an information, that last column is a class in this case. For better visualization, I moved class column as a first, and set the name of other columns to feature_n. our task here is to perform binnary classification. I don't know if we are able to tell something more here.

|   class |   feature_1 |   feature_2 |   feature_3 |   feature_4 |   feature_5 |   feature_6 |   feature_7 |   feature_8 |   feature_9 |   feature_10 |   feature_11 |   feature_12 |   feature_13 |   feature_14 |   feature_15 |   feature_16 |   feature_17 |   feature_18 |   feature_19 |
|--------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
|       1 |      966663 |        6478 |       13397 |      506713 |       14426 |           1 |           0 |           0 |           0 |            0 |            6 |            0 |            0 |            0 |           22 |            1 |          939 |            0 |            0 |
|       0 |       23313 |        6478 |       11001 |      526438 |       18556 |           2 |           0 |           0 |           0 |            0 |            6 |            0 |            0 |            0 |            2 |            1 |          807 |            0 |            0 |
|       0 |      204100 |        6478 |       11001 |      470169 |        2386 |           2 |           0 |           0 |           0 |            0 |            6 |            0 |            0 |            0 |            2 |            1 |          807 |            0 |            0 |
|       1 |      493724 |        6478 |       23327 |      510129 |       63298 |           1 |           0 |           0 |           0 |            0 |            6 |            0 |            0 |            0 |           22 |            1 |          939 |            0 |            0 |
|       1 |      960067 |        6478 |        4678 |      334380 |       14683 |           1 |           0 |       12967 |           1 |            1 |            6 |            0 |            0 |            0 |           22 |           23 |          612 |            0 |            0 |
|       1 |      745958 |        6478 |       19937 |      260034 |       14683 |           1 |           0 |           0 |           0 |            0 |            6 |            0 |            0 |            0 |           22 |            1 |          939 |            0 |            0 |
|       0 |      647157 |        6478 |       11001 |      568372 |       29412 |           2 |           0 |           0 |           0 |            0 |            6 |            0 |            0 |            0 |            2 |            1 |          807 |            0 |            0 |
|       1 |       96464 |        6478 |       15100 |      293436 |       63298 |           1 |           0 |           0 |           0 |            0 |            6 |            0 |            0 |            0 |           22 |            1 |          939 |            0 |            0 |
|       1 |      145426 |        6478 |       10737 |      311835 |       63298 |           1 |           0 |           0 |           0 |            0 |            6 |            0 |            0 |            0 |           22 |            1 |          939 |            0 |            0 |
|       1 |      158427 |        6478 |       18760 |      501746 |       14426 |           1 |           0 |           0 |           0 |            0 |            6 |            0 |            0 |            0 |           22 |            1 |          939 |            0 |            0 |
