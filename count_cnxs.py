

def count_cnxs(dipcp_flows):
    count = {}
    for fields in dipcp_flows:
        cnx = (fields['FlowIPSource'], fields['FlowPortSource'],
               fields['FlowIPDest'], fields['FlowPortDest']) 
        count[cnx] = 1 + count.get(cnx, 0)
    return count
