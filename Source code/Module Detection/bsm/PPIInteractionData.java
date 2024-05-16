package bsm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import org.apache.commons.math3.distribution.NormalDistribution;

import bsm.core.Correlationpro;
import bsm.core.DataSetCore;
import bsm.core.Pearsonr;

public class PPIInteractionData {

	HashMap<Integer, List<Integer>> ppiindex;
	HashMap<Integer, List<Double>> ppivalue;
	
	public PPIInteractionData(String tfBindingDataFile, List<String> finallist) throws IOException {
		ppiindex = new HashMap<Integer, List<Integer>>();
		ppivalue = new HashMap<Integer, List<Double>>();
		String dataFiles = tfBindingDataFile;
		
		HashMap<String, Integer> gene2int = new HashMap<String, Integer>();
		for(int i=0;i<finallist.size();i++) gene2int.put(finallist.get(i), i);

			if (dataFiles != null && !dataFiles.equals("")) {
				BufferedReader br = null;
				try {//���ļ�
					br = new BufferedReader(new InputStreamReader(
							new GZIPInputStream(new FileInputStream(dataFiles))));
				} catch (IOException ex) {
					br = new BufferedReader(new FileReader(dataFiles));
				}

				String szLine = br.readLine();
				StringTokenizer st = new StringTokenizer(szLine, "\t");
				String szh1 = "";

				int numType;
				if (szLine == null) {
					throw new IllegalArgumentException("Empty PPI interaction input file found!");
				} else if (szLine.startsWith("\t")) {
					numType = st.countTokens();
				} else {
					numType = st.countTokens() - 1;
					szh1 = st.nextToken();
				}
				// Read the top row of the file in      //������
				String[] tempRegNames = new String[numType];
				for (int nRegIndex = 0; st.hasMoreTokens(); nRegIndex++) {
					tempRegNames[nRegIndex] = st.nextToken();
				}
				
				while ((szLine = br.readLine()) != null) {
					st = new StringTokenizer(szLine, "\t");
					String szprotein1 = st.nextToken().toUpperCase(Locale.ENGLISH);
					String szprotein2 = st.nextToken().toUpperCase(Locale.ENGLISH);
					double ninput;
					if (st.hasMoreTokens()) {
						String szToken = st.nextToken();
						try {
							ninput = Double.parseDouble(szToken);
						} catch (NumberFormatException nfex) {
							throw new IllegalArgumentException(szToken + " is not a"
									+ " valid score for a direct interaction");
						}
					} else {
						ninput = 1.0;
					}
					
					Integer protein1 = gene2int.get(szprotein1);
					Integer protein2 = gene2int.get(szprotein2);
					
					if(protein1 != null && protein2 != null && protein1 != protein2){
						List<Integer> prolist1 = ppiindex.get(protein1);
						List<Double> vallist1 = ppivalue.get(protein1);
						List<Integer> prolist2 = ppiindex.get(protein2);
						List<Double> vallist2 = ppivalue.get(protein2);
							
							if(prolist1 == null){
								prolist1 = new ArrayList<Integer>();
								vallist1 = new ArrayList<Double>();
								prolist1.add(protein2);
								vallist1.add(ninput);
								ppiindex.put(protein1, prolist1);
								ppivalue.put(protein1, vallist1);
							}else{
								if(!prolist1.contains(protein2)){
									prolist1.add(protein2);
									vallist1.add(ninput);
									ppiindex.put(protein1, prolist1);
									ppivalue.put(protein1, vallist1);
								}
							}
							if(prolist2 == null){
								prolist2 = new ArrayList<Integer>();
								vallist2 = new ArrayList<Double>();
								prolist2.add(protein1);
								vallist2.add(ninput);
								ppiindex.put(protein2, prolist2);
								ppivalue.put(protein2, vallist2);
							}else{
								if(!prolist2.contains(protein1)){
									prolist2.add(protein1);
									vallist2.add(ninput);
									ppiindex.put(protein2, prolist2);
									ppivalue.put(protein2, vallist2);
								}
							}
					}
				}
			}
	}
	
	public void coexpression(BSM_DataSet theds, NormalDistribution Normalcorr, double mincorr,
			HashMap<Integer, List<List<Integer>>> PPIindex, HashMap<Integer, List<Double[]>> PPIvalue) {
		Pearsonr pr = new Pearsonr();
		Correlationpro corpro = new Correlationpro();
		Collection<Integer> cl = ppiindex.keySet();
	    Iterator<Integer> itr = cl.iterator();
	    while (itr.hasNext()) {
	    	int key = itr.next();
	    	double[] exp1 = theds.controlnorm[key];
	    	List<Integer> genes = ppiindex.get(key);
	    	List<Double> bindings = ppivalue.get(key);
	    	if(genes.size() > 0) {
	    		List<Integer> genes2 = new ArrayList<Integer>();
	    		List<Double> value = new ArrayList<Double>();
	    		for(int i=0;i<genes.size();i++) {
	    			double[] exp2 = theds.controlnorm[genes.get(i)];
	    			double corr = pr.cosineSimilarity(exp1, exp2);
	    			if((bindings.get(i) > 0 && corr > mincorr) || (bindings.get(i) < 0 && corr < -mincorr)
	    					|| (bindings.get(i) == 0 && Math.abs(corr) > mincorr)) {
	    				value.add(corpro.getProportion(Normalcorr, corr, mincorr)[1]);
	    				genes2.add(genes.get(i));
	    			}
	    		}
	    		if(value.size() > 1) {
	    			List<List<Integer>> current = PPIindex.get(key);
	    			if(current != null && current.size()>0) {
	    				if(!contain(current, genes2)) {
	    					current.add(genes2);
	    					Double[] weight = new Double[value.size()];
			    			double sum=0;
			    			for(int i=0;i<value.size();i++) {
			    				weight[i] = Math.exp(value.get(i));
			    				sum += weight[i];
			    			}
			    			for(int i=0;i<weight.length;i++) {
			    				weight[i] /= sum;
			    			}
			    			List<Double[]> currentvalue = PPIvalue.get(key);
			    			currentvalue.add(weight);
			    			PPIindex.put(key, current);
			    			PPIvalue.put(key, currentvalue);
	    				}
	    				
	    			} else {
	    				Double[] weight = new Double[value.size()];
		    			double sum=0;
		    			for(int i=0;i<value.size();i++) {
		    				weight[i] = Math.exp(value.get(i));
		    				sum += weight[i];
		    			}
		    			for(int i=0;i<weight.length;i++) {
		    				weight[i] /= sum;
		    			}
		    			List<List<Integer>> list3 = new ArrayList<List<Integer>>();
		    			list3.add(genes2);
		    			List<Double[]> value3 = new ArrayList<Double[]>();
		    			value3.add(weight);
		    			PPIindex.put(key, list3);
		    			PPIvalue.put(key, value3);
	    			}
	    		}
	    	}
	    }
	}
	
	private boolean contain (List<List<Integer>> all, List<Integer> list1) {
		List<Integer> list2 = new ArrayList<Integer>(list1);
		Collections.sort(list2);
		for(int i=0;i<all.size();i++) {
			List<Integer> list = new ArrayList<Integer>(all.get(i));
			Collections.sort(list);
			if(list.toString().equals(list2.toString())) {
				return true;
			}
		}
		return false;
	}
	
}
