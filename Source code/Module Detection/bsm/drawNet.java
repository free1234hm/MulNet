package bsm;

import edu.uci.ics.jung.algorithms.layout.KKLayout;
import edu.uci.ics.jung.graph.SparseMultigraph;
import edu.uci.ics.jung.graph.util.EdgeType;
import edu.uci.ics.jung.visualization.VisualizationViewer;
import edu.uci.ics.jung.visualization.control.DefaultModalGraphMouse;
import edu.uci.ics.jung.visualization.control.GraphMouseListener;
import edu.uci.ics.jung.visualization.control.ModalGraphMouse;
import edu.uci.ics.jung.visualization.decorators.ToStringLabeller;
import edu.uci.ics.jung.visualization.renderers.Renderer.VertexLabel.Position;
import edu.uci.ics.jung.visualization.util.VertexShapeFactory;
import edu.uci.ics.jung.algorithms.layout.FRLayout;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.Paint;
import java.awt.Shape;
import java.awt.Stroke;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTable;
import javax.swing.SpinnerNumberModel;
import javax.swing.border.EmptyBorder;
import org.apache.commons.collections15.Transformer;
import bsm.core.TableModelST;
import bsm.core.TableSorter;


public class drawNet {
	private JPanel contentPane;
	private JTable table;
	private JTable table2;
	TableSorter sorter;
	TableSorter sorter2;
	private JFrame frame;
	private JFrame saveImageFrame;
	private JFrame saveNetworkFrame;
	private JScrollPane scrollPane;
	private Set<String> listTF = new HashSet<String>();
	private Set<String> listTG = new HashSet<String>();
	private String[] Names = {"Regulator", "Num target"};
	private String[] Names2 = { "Gene", "Num neighbor"};
	SparseMultigraph<String, String> initialgraph = null;
	public void drawPic(Set<String> directnetwork, HashMap<String, Set<String>> tftg, BSM_DataSet rawDataSet) {
		// TODO Auto-generated method stub
		initialgraph = new SparseMultigraph<String, String>();
		SparseMultigraph<String, String> graph = new SparseMultigraph<String, String>();
		KKLayout<String, String> layout = new KKLayout<String, String>(graph);
		//FRLayout<String, String> layout = new FRLayout<String, String>(graph);
		VisualizationViewer<String, String> vv = new VisualizationViewer<String, String>(layout);
		listTF = tftg.keySet();
		
		for(Entry<String, Set<String>> entry:tftg.entrySet())
		  {
			String tf = entry.getKey();
			Set<String> tg = tftg.get(tf);
			for(String g:tg) {
				if(!listTF.contains(g)) {
					listTG.add(g);
				}
				initialgraph.addVertex(tf);
				initialgraph.addVertex(g);
				initialgraph.addEdge(tf+"\t"+g, tf, g, EdgeType.DIRECTED);
				graph.addVertex(tf);
				graph.addVertex(g);
				graph.addEdge(tf+"\t"+g, tf, g, EdgeType.DIRECTED);
			}
		  }
		
		String[][] TFlist = new String[listTF.size()][2];
		int count=0;
		for(String tf:listTF) {
			TFlist[count][0] = tf;
			TFlist[count][1] = initialgraph.getNeighborCount(tf)+"";
			count++;
		}
		
		for(String tt:directnetwork){
			String[] ppi = tt.split("\t");
			if(!listTF.contains(ppi[0])) {
				listTG.add(ppi[0]);
			}
			if(!listTF.contains(ppi[1])) {
				listTG.add(ppi[1]);
			}
			initialgraph.addVertex(ppi[0]);
			initialgraph.addVertex(ppi[1]);
			initialgraph.addEdge(ppi[0]+"\t"+ppi[1], ppi[0], ppi[1], EdgeType.UNDIRECTED);
			graph.addVertex(ppi[0]);
			graph.addVertex(ppi[1]);
			graph.addEdge(ppi[0]+"\t"+ppi[1], ppi[0], ppi[1], EdgeType.UNDIRECTED);
		}
		
		String[][] Genelist = new String[listTG.size()][2];
		int count2 = 0;
		for(String tg:listTG) {
			Genelist[count2][0] = tg;
			Genelist[count2][1] = initialgraph.getNeighborCount(tg)+"";
			count2++;
		}
		
		//初始化点颜色
		Transformer<String, Paint> vertexPaint = new Transformer<String, Paint>() {
			public Paint transform(String s) {
				if(listTF.contains(s)){
					return Color.GREEN;
				} else if(listTG.contains(s)){
					return Color.PINK;
				} else {					
					return null;
				}
			}
		};
		//初始化边颜色
		Transformer<String, Paint> edgePaint = new Transformer<String, Paint>() {
			public Paint transform(String s) {
				if(initialgraph.getEdgeType(s) == EdgeType.UNDIRECTED) {
					return Color.BLUE;
				} else if(initialgraph.getEdgeType(s) == EdgeType.DIRECTED) {
					return Color.BLACK;
				} else {
					return null;
				}
			}
		};
		
		/*
		//点击节点后修改边框颜色
		Transformer<String, Paint> changePaint = new Transformer<String, Paint>() {
			public Paint transform(String s) {
				if(getgeneList.contains(s)){
					return Color.RED;
				}
				else {					
					return Color.BLACK;
				}
			}
		};
		//原始边框粗细
		Transformer<String, Stroke> changeStroke1 = new Transformer<String, Stroke>() {
			public Stroke transform(String s) {
				if(getgeneList.contains(s)){
					return new BasicStroke(1f);
				}
				else {					
					return null;
				}
			}
		};
		//修改边框粗细
		Transformer<String, Stroke> changeStroke3 = new Transformer<String, Stroke>() {
			public Stroke transform(String s) {
				if(getgeneList.contains(s)){
					return new BasicStroke(2f);
				}
				else {					
					return null;
				}
			}
		};
		*/
		
		//顶点变成圆角矩形框
				Transformer<String, Integer> vst1 = new Transformer<String, Integer>() {
					public Integer transform(String s) {
						int len = s.length();
						if (len < 5) len = 5;
						return new Integer(len * 8 + 7);
					}
				};
				Transformer<String, Integer> vst2 = new Transformer<String, Integer>() {
					public Integer transform(String s) {
						int len = s.length();
						if (len < 5)
							len = 5;
						return new Integer(len * 9 + 7);
					}
				};
				Transformer<String, Float> vart = new Transformer<String, Float>() {
					public Float transform(String s) {
						int len = s.length();
						if (len < 5) {
							return new Float(0.4);
						} else {
							return new Float(Math.max(0.2, (double)2/len));
						}
					}
				};
				VertexShapeFactory<String> vsf1 = new VertexShapeFactory<String>(vst1, vart);
				VertexShapeFactory<String> vsf2 = new VertexShapeFactory<String>(vst2, vart);
				Transformer<String, Shape> vstr = new Transformer<String, Shape>() {
					public Shape transform(String s) {
						if(listTF.contains(s)){
							return vsf2.getRegularStar(s, 8);
						} else if(listTG.contains(s)){
							return vsf1.getRoundRectangle(s);
						} else {					
							return null;
						}
						
					}
				};
		
		//图形变换模式
		DefaultModalGraphMouse<Integer, String> gm = new DefaultModalGraphMouse<Integer, String>();		
		gm.setMode(ModalGraphMouse.Mode.PICKING);	
		vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint); //fill color
		vv.getRenderContext().setEdgeDrawPaintTransformer(edgePaint);	
		vv.getRenderContext().setVertexStrokeTransformer(new Transformer<String, Stroke>() {//边框宽度
	        public Stroke transform(String v) {
	        	return new BasicStroke(0);
	        }
	    });
		vv.getRenderContext().setVertexLabelTransformer(new ToStringLabeller<String>()); 
		vv.getRenderContext().setVertexShapeTransformer(vstr); //shape
		vv.getRenderer().getVertexLabelRenderer().setPosition(Position.CNTR);
		vv.setGraphMouse(gm);
		vv.setBackground(Color.WHITE);
		/*
		vv.getRenderContext().setVertexDrawPaintTransformer(new Transformer<String, Paint>() {//边框颜色
	        public Paint transform(String v) {
	            if (vv.getPickedVertexState().isPicked(v)) {
	                return Color.CYAN;
	            } else {
	                return Color.BLACK;
	            }
	        }
	    });
	    
		vv.getRenderContext().setVertexStrokeTransformer(new Transformer<String, Stroke>() {//边框宽度
	        public Stroke transform(String v) {
	            if (vv.getPickedVertexState().isPicked(v)) {
	                return new BasicStroke(2f);
	            } else {
	            	return new BasicStroke(1f);
	            }
	        }
	    });
		*/
		
		//设置对话框属性
		frame = new JFrame("Network diagram");
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setBounds(10, 10, 1350, 700);
		
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		frame.setContentPane(contentPane);
		
		//添加table单击事件
		sorter = new TableSorter(new TableModelST(TFlist, Names));
		table = new JTable(sorter);
		sorter.setTableHeader(table.getTableHeader());
		
		table.addMouseListener(new MouseAdapter() {
		     public void mouseClicked(MouseEvent e) { 
		    	 if(table.getValueAt(table.getSelectedRow(),0)!=null && initialgraph.getVertexCount()>0){
		    		 for(String str : initialgraph.getVertices())graph.removeVertex(str);
		    		 for(String str : initialgraph.getEdges()) graph.removeEdge(str);
		    		 
		    		 String TFID = (String)table.getValueAt(table.getSelectedRow(), 0);
		    		 Collection<String> genes = initialgraph.getNeighbors(TFID);
		    		 graph.addVertex(TFID);
		    		 if(genes!=null && genes.size()>0){
		    			 for(String str : genes) {
		    				 graph.addVertex(str);
		    			 }	 
		    			 Collection<String> edges = initialgraph.getIncidentEdges(TFID);
		    			 for(String edge : edges) {
		    				 graph.addEdge(edge, edge.split("\t")[0], edge.split("\t")[1], initialgraph.getEdgeType(edge));
		    			 }
		    		 }
		    		 vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint);
		    		 vv.getRenderContext().setEdgeDrawPaintTransformer(edgePaint);
		    		 scrollPane.repaint();
		    		 scrollPane.updateUI(); 
		    	 }
		     }
		});
		
		//添加table2单击事件
		sorter2 = new TableSorter(new TableModelST(Genelist, Names2));
		table2 = new JTable(sorter2);
		sorter2.setTableHeader(table2.getTableHeader());
		
		table2.addMouseListener(new MouseAdapter() {
		     public void mouseClicked(MouseEvent e) { 
		    	 if(table2.getValueAt(table2.getSelectedRow(),0)!=null && initialgraph.getVertexCount()>0){
		    		 
		    		 for(String str : initialgraph.getVertices())graph.removeVertex(str);
		    		 for(String str : initialgraph.getEdges()) graph.removeEdge(str);
		    		 
		    		 String TFID = (String)table2.getValueAt(table2.getSelectedRow(), 0);
		    		 Collection<String> genes = initialgraph.getNeighbors(TFID);
		    		 graph.addVertex(TFID);
		    		 if(genes != null && genes.size()>0){
		    			 for(String str : genes) {
		    				 graph.addVertex(str);
		    			 }
		    			 Collection<String> edges = initialgraph.getIncidentEdges(TFID);
		    			 for(String edge : edges) {
		    				 graph.addEdge(edge, edge.split("\t")[0], edge.split("\t")[1], initialgraph.getEdgeType(edge));
		    			 }
		    		 }
		    		 vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint);
		    		 vv.getRenderContext().setEdgeDrawPaintTransformer(edgePaint);
		    		 scrollPane.repaint();
		    		 scrollPane.updateUI();
		    	 }
		     }
		});
		
		
		Container c = frame.getContentPane();
		GroupLayout mainLayout = new GroupLayout(c);
		
		JScrollPane scrollPane1 = new JScrollPane(table);
		c.add(scrollPane1);
		//scrollPane1.setViewportView(table);
		
		JScrollPane scrollPane12 = new JScrollPane(table2);
		c.add(scrollPane12);
		//scrollPane12.setViewportView(table2);
		
		
		scrollPane = new JScrollPane(vv);
		scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
		scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_NEVER);
		//panel.add(scrollPane2); 
		c.add(scrollPane);
		
		JLabel degree1 = new JLabel("Degree :");
		SpinnerNumberModel degreevalue = new SpinnerNumberModel(new Integer(10), new Integer(0), null, new Integer(1));
		JSpinner degree2 = new JSpinner(degreevalue);
		degree2.setMaximumSize(new Dimension(120,30));
		JButton filter = new JButton();
		filter.setText("Filter regulators");
		c.add(filter);
		
		JButton restore = new JButton();
		restore.setText("Restore");
		c.add(restore);
		JButton savemap = new JButton();
		savemap.setText("Save Image");
		c.add(savemap);
		JButton savenetwork = new JButton();
		savenetwork.setText("Save Network File");
		c.add(savenetwork);
		
		filter.addActionListener(e -> {
			int threshold = (int) degree2.getValue();
			for(String str : initialgraph.getVertices()) graph.removeVertex(str);
			for(String str : initialgraph.getEdges()) graph.removeEdge(str);
			
			for(String str : initialgraph.getVertices()) {
				if(tftg.get(str)!=null && tftg.get(str).size()>=threshold){
					Set<String> neighbors = new HashSet<String>();
					Collection<String> edges = initialgraph.getIncidentEdges(str);
	    			 for(String edge : edges) {
	    				 if(initialgraph.getEdgeType(edge)==EdgeType.DIRECTED && str.equals(edge.split("\t")[0])) {
	    					 neighbors.add(edge.split("\t")[0]);
	    					 neighbors.add(edge.split("\t")[1]);
	    					 graph.addEdge(edge, edge.split("\t")[0], edge.split("\t")[1], initialgraph.getEdgeType(edge));
	    				 } else if(initialgraph.getEdgeType(edge)==EdgeType.UNDIRECTED) {
	    					 neighbors.add(edge.split("\t")[0]);
	    					 neighbors.add(edge.split("\t")[1]);
	    					 graph.addEdge(edge, edge.split("\t")[0], edge.split("\t")[1], initialgraph.getEdgeType(edge));
	    				 }
	    			 }
	    			 graph.addVertex(str);
					 for(String str2 : neighbors) {
						 graph.addVertex(str2);
					 }
				}
			}			
			vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint);
			vv.getRenderContext().setEdgeDrawPaintTransformer(edgePaint);
			scrollPane.repaint();
			scrollPane.updateUI();
        });
		
		restore.addActionListener(e -> {
			for(String str : initialgraph.getVertices()) graph.removeVertex(str);
			for(String str : initialgraph.getEdges()) graph.removeEdge(str);
			for(String str : initialgraph.getVertices()) graph.addVertex(str);
			for(String str : initialgraph.getEdges()) graph.addEdge(str, str.split("\t")[0], str.split("\t")[1], initialgraph.getEdgeType(str));
			vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint); //fill color
			vv.getRenderContext().setVertexDrawPaintTransformer(edgePaint);
			scrollPane.repaint();
			scrollPane.updateUI();
        });
		
		savemap.addActionListener(e -> {
			javax.swing.SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					if (saveImageFrame == null) {
						saveImageFrame = new JFrame("Save as Image");
						saveImageFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
						saveImageFrame.setLocation(400,300);
						TSMinerGui_SaveImage2 newContentPane = new TSMinerGui_SaveImage2(saveImageFrame,scrollPane);
						newContentPane.setOpaque(true);
						// content panes must be opaque
						saveImageFrame.setContentPane(newContentPane);
						// Display the window.
						saveImageFrame.pack();
					} else {
						saveImageFrame.setExtendedState(Frame.NORMAL);
					}
					saveImageFrame.setVisible(true);
				}
			});
			
        });
		
		savenetwork.addActionListener(e -> {
			javax.swing.SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					if (saveNetworkFrame == null) {
						saveNetworkFrame = new JFrame("Save Network File");
						saveNetworkFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
						saveNetworkFrame.setLocation(400,300);
						SaveNetwork newContentPane = new SaveNetwork(saveNetworkFrame, graph);
						newContentPane.setOpaque(true);
						// content panes must be opaque
						saveNetworkFrame.setContentPane(newContentPane);
						// Display the window.
						saveNetworkFrame.pack();
					} else {
						saveNetworkFrame.setExtendedState(Frame.NORMAL);
					}
					saveNetworkFrame.setVisible(true);
				}
			});
        });
		
		GroupLayout.SequentialGroup  DEanalysis1 = mainLayout.createSequentialGroup().addGap(30);
		GroupLayout.SequentialGroup  TFfilter1 = mainLayout.createSequentialGroup().addComponent(degree1).addGap(3).addComponent(degree2).addComponent(filter);
		GroupLayout.SequentialGroup  hParalGroup01 = mainLayout.createSequentialGroup().addGroup(DEanalysis1).addGap(20).addGroup(TFfilter1).addGap(20).addComponent(restore).addGap(5);
		GroupLayout.SequentialGroup hParalGroup02 = mainLayout.createSequentialGroup().addComponent(savemap).addGap(10).addComponent(savenetwork).addGap(5);
		GroupLayout.ParallelGroup hParalGroup03 = mainLayout.createParallelGroup(Alignment.TRAILING).addComponent(scrollPane, GroupLayout.DEFAULT_SIZE, 646, Short.MAX_VALUE)
				.addGroup(hParalGroup01);
		GroupLayout.ParallelGroup hParalGroup04 = mainLayout.createParallelGroup(Alignment.TRAILING).addComponent(scrollPane1, GroupLayout.DEFAULT_SIZE, 400, Short.MAX_VALUE)
				.addComponent(scrollPane12, GroupLayout.DEFAULT_SIZE, 400, Short.MAX_VALUE).addGroup(hParalGroup02);
		GroupLayout.SequentialGroup  hParalGroup = mainLayout.createSequentialGroup().addContainerGap().addGroup(hParalGroup03).addGap(15).addGroup(hParalGroup04).addContainerGap();
		mainLayout.setHorizontalGroup(hParalGroup);
		
		GroupLayout.ParallelGroup  DEanalysis2 = mainLayout.createParallelGroup();
		GroupLayout.ParallelGroup  TFfilter2 = mainLayout.createParallelGroup().addComponent(degree1).addComponent(degree2).addComponent(filter);
		GroupLayout.ParallelGroup hParalGroup05 = mainLayout.createParallelGroup().addGroup(DEanalysis2).addGroup(TFfilter2).addComponent(restore);
		GroupLayout.ParallelGroup hParalGroup06 = mainLayout.createParallelGroup().addComponent(savemap).addComponent(savenetwork);
		GroupLayout.SequentialGroup hParalGroup07 = mainLayout.createSequentialGroup().addContainerGap().addComponent(scrollPane, GroupLayout.DEFAULT_SIZE, 646, Short.MAX_VALUE)
				.addGap(10).addGroup(hParalGroup05).addGap(10);
		GroupLayout.SequentialGroup hParalGroup08 = mainLayout.createSequentialGroup().addContainerGap().addComponent(scrollPane1, GroupLayout.DEFAULT_SIZE, 400, Short.MAX_VALUE)
				.addGap(10).addComponent(scrollPane12, GroupLayout.DEFAULT_SIZE, 400, Short.MAX_VALUE).addGap(10).addGroup(hParalGroup06).addGap(10);
		GroupLayout.ParallelGroup  hParalGroup2 = mainLayout.createParallelGroup().addGroup(hParalGroup07).addGap(18).addGroup(hParalGroup08);
		mainLayout.setVerticalGroup(hParalGroup2);

		//panel.setLayout(new BorderLayout(0, 0));
		contentPane.setLayout(mainLayout);
		frame.setVisible(true);
	}

}
