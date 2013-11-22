#!/usr/bin/env python
"Module to test metrics from tstat and dipcp regarding out of order packets"

from scapy.all import IP, TCP, wrpcap
import sys

#SEQ_NBS = [0, 1500, 3000, 7500, 4500, 6000, 3000]
SEQ_NBS = [0,  500, 1000, 1500, 3000,  500, 1000, 4500, 9000, 6000, 7500, 10500]
SIZES = [500, 500, 500, 1500, 1500, 500, 500, 1500, 1500, 1500, 1500, 1500]
#PAYLOAD = 'X' * 1500

def main():
    "Send packets with specific seq nbs"
    packet_list = [IP(src='0.0.0.1',
                      dst='1.0.0.1')/TCP(dport=10001,sport=10002,
                                         flags="S",ack=0, seq=0)]
    packet_list.append(IP(src='1.0.0.1',
                          dst='0.0.0.1')/TCP(dport=10002,sport=10001,
                                             flags="SA",ack=1, seq=0))
    packet_list.append(IP(src='0.0.0.1',
                          dst='1.0.0.1')/TCP(dport=10001,sport=10002,
                                             flags="A",ack=1, seq=1))
    for nb, size in zip(SEQ_NBS, SIZES):
        packet_list.append(IP(src='0.0.0.1',
                              dst='1.0.0.1')/TCP(dport=10001,sport=10002,
                                                 flags="A",ack=1,
                                                 seq=nb+1)/('X' * size))
    wrpcap("packet_list.dmp", packet_list)

if __name__ == "__main__":
    sys.exit(main())
