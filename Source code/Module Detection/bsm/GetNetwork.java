package bsm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import bsm.LearningCase.Casenode;
import bsm.core.Pearsonr;

public class GetNetwork {
	public void network(Casenode node, BSM_DataSet theds, List<RegulatorBindingData> KnownModules, 
			List<PPIInteractionData> ppiData, double mincorr) {
		Pearsonr pr = new Pearsonr();
		List<Integer> genelsit = node.genelist;
		node.directnetwork = new HashSet<String>();
		node.coexpressionnetwork = new HashSet<String>();
		node.tftg = new HashMap<String, Set<String>>();
		for(int i=0;i<genelsit.size();i++) {
			int gene1 = genelsit.get(i);
			double[] exp1 = theds.controlnorm[gene1];
			String g1 = theds.genenames[gene1];
			for(int j=i+1;j<genelsit.size();j++) {
				int gene2 = genelsit.get(j);
			    String g2 = theds.genenames[gene2];
			    boolean found = false;
			    if(KnownModules != null) {
			    	for(int m=0;m<KnownModules.size();m++) {
						RegulatorBindingData bindingData = KnownModules.get(m);
						int[] tf1 = bindingData.gene2RegBindingIndex[gene1];
						int[] tf2 = bindingData.gene2RegBindingIndex[gene2];
						List<Integer> overlap = overlap(tf1, tf2);
							for(int tf:overlap) {
	    						if(node.tftg.get(bindingData.regNames[tf]) != null) {
	    							Set<String> aa = node.tftg.get(bindingData.regNames[tf]);
	    							if(!aa.contains(g1)) aa.add(g1);
	    							if(!aa.contains(g2)) aa.add(g2);
	    							node.tftg.put(bindingData.regNames[tf], aa);
	    						} else {
	    							Set<String> aa = new HashSet<String>();
	    							aa.add(g1);
	    							aa.add(g2);
	    							node.tftg.put(bindingData.regNames[tf], aa);
	    						}
	    					}
					}
			    }
			    if(ppiData!=null) {
			    	 for(int m=0;m<ppiData.size();m++) {
					    	PPIInteractionData ppi = ppiData.get(m);
					    	List<Integer> geneset = ppi.ppiindex.get(gene1);
					    	if(geneset!=null && geneset.size()>0) {
					    		if(geneset.contains(gene2)) {
					    			String value = g1+"\t"+g2+"\t"+"direct";
									node.directnetwork.add(value);
									found = true;
									break;
					    		}
					    	}
					    }
			    }
			   
			    if(!found) {
					double[] exp2 = theds.controlnorm[gene2];
					double corr = pr.cosineSimilarity(exp1, exp2);
					if(Math.abs(corr) >= mincorr) {
						node.coexpressionnetwork.add(g1+"\t"+g2+"\t"+"co-expression");
					}
				}
			}
		}
	}
	
	public List<String> network(List<Integer> genelsit, BSM_DataSet theds, double mincorr,
			List<RegulatorBindingData> tfdata, List<PPIInteractionData> PPIdata) {
		List<String> networkfile = new ArrayList<String>();
		List<String> g1g2 = new ArrayList<String>();
		List<String> g1g3 = new ArrayList<String>();
		for(int i=0;i<genelsit.size();i++) {
			int gene1 = genelsit.get(i);
			String g1 = theds.genenames[gene1];
			for(int m=0;m<tfdata.size();m++) {
				RegulatorBindingData tf1 = tfdata.get(m);
				for(int n=0;n<tf1.gene2RegBindingIndex[gene1].length;n++) {
					int[] tglist = tf1.reg2GeneBindingIndex[tf1.gene2RegBindingIndex[gene1][n]];
					for(int gene2:tglist) {
						if(gene2!=gene1 && genelsit.contains(gene2)) {
							String g2 = theds.genenames[gene2];
							if(!g1g2.contains(g1+"_"+g2)) {
								g1g2.add(g1+"_"+g2);
								g1g2.add(g2+"_"+g1);
								double[] exp1 = theds.controlnorm[gene1];
								double[] exp2 = theds.controlnorm[gene2];
								Pearsonr pr = new Pearsonr();
								double corr = pr.cosineSimilarity(exp1, exp2);
								if(Math.abs(corr) >= mincorr) {
									networkfile.add(g1+"\t"+g2+"\t"+"co-regulation");
								}
							}
						}
					}
				}
			}
			for(int m=0;m<PPIdata.size();m++) {
				PPIInteractionData ppi = PPIdata.get(m);
				List<Integer> tgs = ppi.ppiindex.get(gene1);
				if(tgs != null) {
					for(int j=i+1;j<genelsit.size();j++) {
						int gene2 = genelsit.get(j);
						String g2 = theds.genenames[gene2];
						if(tgs.contains(gene2)) {
							g1g3.add(g1+"_"+g2);
							double[] exp1 = theds.controlnorm[gene1];
							double[] exp2 = theds.controlnorm[gene2];
							Pearsonr pr = new Pearsonr();
							double corr = pr.cosineSimilarity(exp1, exp2);
							if(Math.abs(corr) >= mincorr) {
								networkfile.add(g1+"\t"+g2+"\t"+"direct");
							}
							break;
						}
					}
				}
			}
		}
		
		for(int i=0;i<genelsit.size();i++) {
			int gene1 = genelsit.get(i);
			String g1 = theds.genenames[gene1];
			for(int j=i+1;j<genelsit.size();j++) {
				int gene2 = genelsit.get(j);
				String g2 = theds.genenames[gene2];
				if(!g1g2.contains(g1+"_"+g2) && !g1g3.contains(g1+"_"+g2)) {
						double[] exp1 = theds.controlnorm[gene1];
						double[] exp2 = theds.controlnorm[gene2];
						Pearsonr pr = new Pearsonr();
						double corr = pr.cosineSimilarity(exp1, exp2);
						if(Math.abs(corr) >= mincorr) {
							String value = g1+"\t"+g2+"\t"+"co-expression";
							networkfile.add(value);
						}
				}
			}
		}
		return networkfile;
	}
	public List<String> network(List<Integer> genelsit, List<BSM_DataSet> theds, double mincorr,
			List<RegulatorBindingData> tfdata, List<PPIInteractionData> PPIdata) {
		BSM_DataSet data0 = theds.get(0);
		List<String> networkfile = new ArrayList<String>();
		List<String> g1g2 = new ArrayList<String>();
		List<String> g1g3 = new ArrayList<String>();
		for(int i=0;i<genelsit.size();i++) {
			int gene1 = genelsit.get(i);
			String g1 = data0.genenames[gene1];
			for(int m=0;m<tfdata.size();m++) {
				RegulatorBindingData tf1 = tfdata.get(m);
				for(int n=0;n<tf1.gene2RegBindingIndex[gene1].length;n++) {
					int[] tglist = tf1.reg2GeneBindingIndex[tf1.gene2RegBindingIndex[gene1][n]];
					for(int gene2:tglist) {
						if(gene2!=gene1 && genelsit.contains(gene2)) {
							String g2 = data0.genenames[gene2];
							if(!g1g2.contains(g1+"_"+g2)) {
								g1g2.add(g1+"_"+g2);
								g1g2.add(g2+"_"+g1);
								boolean sig = true;
								for(BSM_DataSet data:theds) {
									double[] exp1 = data.controlnorm[gene1];
									double[] exp2 = data.controlnorm[gene2];
									Pearsonr pr = new Pearsonr();
									double corr = pr.cosineSimilarity(exp1, exp2);
									if(Math.abs(corr) < mincorr) {
										sig = false;
										break;
									}
								}
								if(sig) {
									networkfile.add(g1+"\t"+g2+"\t"+"co-regulation");
								}
							}
						}
					}
				}
			}
			for(int m=0;m<PPIdata.size();m++) {
				PPIInteractionData ppi = PPIdata.get(m);
				List<Integer> tgs = ppi.ppiindex.get(gene1);
				if(tgs != null) {
					for(int j=i+1;j<genelsit.size();j++) {
						int gene2 = genelsit.get(j);
						String g2 = data0.genenames[gene2];
						if(tgs.contains(gene2)) {
							g1g3.add(g1+"_"+g2);
							boolean sig = true;
							for(BSM_DataSet data:theds) {
								double[] exp1 = data.controlnorm[gene1];
								double[] exp2 = data.controlnorm[gene2];
								Pearsonr pr = new Pearsonr();
								double corr = pr.cosineSimilarity(exp1, exp2);
								if(Math.abs(corr) < mincorr) {
									sig = false;
									break;
								}
							}
							if(sig) {
								networkfile.add(g1+"\t"+g2+"\t"+"direct");
							}
							break;
						}
					}
				}
			}
		}
		
		for(int i=0;i<genelsit.size();i++) {
			int gene1 = genelsit.get(i);
			String g1 = data0.genenames[gene1];
			for(int j=i+1;j<genelsit.size();j++) {
				int gene2 = genelsit.get(j);
				String g2 = data0.genenames[gene2];
				if(!g1g2.contains(g1+"_"+g2) && !g1g3.contains(g1+"_"+g2)) {
					boolean sig = true;
					for(BSM_DataSet data:theds) {
						double[] exp1 = data.controlnorm[gene1];
						double[] exp2 = data.controlnorm[gene2];
						Pearsonr pr = new Pearsonr();
						double corr = pr.cosineSimilarity(exp1, exp2);
						if(Math.abs(corr) < mincorr) {
							sig = false;
							break;
						}
					}
					if(sig) {
						String value = g1+"\t"+g2+"\t"+"co-expression";
						networkfile.add(value);
					}
				}
			}
		}
		return networkfile;
	}
	private List<Integer> overlap(int[] list1, int[] list2) {
		List<Integer> list = new ArrayList<Integer>();
		for(int i=0;i<list1.length;i++) {
			for(int j=0;j<list2.length;j++) {
				if(list1[i] == list2[j]) {
					list.add(list1[i]);
				}
			}
		}
		return list;
	}
}
