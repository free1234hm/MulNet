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
import java.util.Random;
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

import bsm.core.SaveNetwork;
import bsm.core.TSMinerGui_SaveImage2;
import bsm.core.TableModelST;
import bsm.core.TableSorter;


public class drawNet {
	private JPanel contentPane;
	private JFrame frame;
	private JFrame saveImageFrame;
	private JFrame saveNetworkFrame;
	private JScrollPane scrollPane;
	private Set<String> listTF = new HashSet<String>();
	SparseMultigraph<String, String> initialgraph = null;
	public void drawPic(List<String> genes, List<String> types, HashMap<Integer, HashMap<String, Set<Integer>>> network, 
			HashMap<String, HashMap<String, Set<String>>> survivalnetwork,
			HashMap<String, List<Integer>> tftg, BSM_DataSet rawDataSet) {
		
		initialgraph = new SparseMultigraph<String, String>();
		SparseMultigraph<String, String> graph = new SparseMultigraph<String, String>();
		KKLayout<String, String> layout = new KKLayout<String, String>(graph);
		//FRLayout<String, String> layout = new FRLayout<String, String>(graph);
		VisualizationViewer<String, String> vv = new VisualizationViewer<String, String>(layout);
		listTF = tftg.keySet();
		for(int i=0;i<genes.size();i++) {
			String gene = genes.get(i);
			String type = types.get(i);
			if(tftg.keySet().contains(type)) {
				initialgraph.addVertex(type);
				initialgraph.addVertex(gene);
				initialgraph.addEdge(type+"\t"+gene, type, gene, EdgeType.DIRECTED);
				graph.addVertex(type);
				graph.addVertex(gene);
				graph.addEdge(type+"\t"+gene, type, gene, EdgeType.DIRECTED);
				HashMap<String, Set<Integer>> neighbor = network.get(rawDataSet.gene2int.get(gene));
				if(neighbor!=null && neighbor.size()>0) {
					Set<Integer> targets = neighbor.get(type);
					if(targets!=null && targets.size()>0) {
						for(int aa:targets) {
							initialgraph.addVertex(type);
							initialgraph.addVertex(rawDataSet.genenames[aa]);
							initialgraph.addEdge(type+"\t"+rawDataSet.genenames[aa], type, rawDataSet.genenames[aa], EdgeType.DIRECTED);
							graph.addVertex(type);
							graph.addVertex(rawDataSet.genenames[aa]);
							graph.addEdge(type+"\t"+rawDataSet.genenames[aa], type, rawDataSet.genenames[aa], EdgeType.DIRECTED);
						}
					}
				}
			} else {
				HashMap<String, Set<Integer>> neighbor = network.get(rawDataSet.gene2int.get(gene));
				if(neighbor!=null && neighbor.size()>0) {
					Set<Integer> targets = neighbor.get(type);
					if(targets!=null && targets.size()>0) {
						for(int aa:targets) {
							initialgraph.addVertex(gene);
							initialgraph.addVertex(rawDataSet.genenames[aa]);
							initialgraph.addEdge(gene+"\t"+rawDataSet.genenames[aa], gene, rawDataSet.genenames[aa], EdgeType.UNDIRECTED);
							graph.addVertex(gene);
							graph.addVertex(rawDataSet.genenames[aa]);
							graph.addEdge(gene+"\t"+rawDataSet.genenames[aa], gene, rawDataSet.genenames[aa], EdgeType.UNDIRECTED);
						}
					} else {
						initialgraph.addVertex(gene);
						graph.addVertex(gene);
					}
				} else {
					initialgraph.addVertex(gene);
					graph.addVertex(gene);
				}
			}
		}
		
		//初始化点颜色
		Transformer<String, Paint> vertexPaint = new Transformer<String, Paint>() {
			public Paint transform(String s) {
				if(listTF.contains(s)){
					return Color.GREEN;
				} else {
					return Color.PINK;
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
		//边框颜色
		Transformer<String, Paint> strokecolor = new Transformer<String, Paint>() {
			public Paint transform(String s) {
				if(genes.contains(s)){
					return Color.RED;
				} else {					
					return Color.BLACK;
				}
			}
		};
		//边框粗细
		Transformer<String, Stroke> strokePaint = new Transformer<String, Stroke>() {
			public Stroke transform(String s) {
				if(genes.contains(s)){
					return new BasicStroke(2f);
				} else {					
					return new BasicStroke(1f);
				}
			}
		};
		
		//顶点变成圆角矩形框
				Transformer<String, Integer> vst1 = new Transformer<String, Integer>() {
					public Integer transform(String s) {
						int len = s.length();
						if (len < 5) len = 5;
						return new Integer(len * 8 + 8);
					}
				};
				Transformer<String, Integer> vst2 = new Transformer<String, Integer>() {
					public Integer transform(String s) {
						int len = s.length();
						if (len < 5)
							len = 5;
						return new Integer(len * 9 + 8);
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
						if(tftg.keySet().contains(s)){
							return vsf2.getRegularStar(s, 8);
						} else {
							return vsf1.getRoundRectangle(s);
						}
					}
				};
		
		//图形变换模式
		DefaultModalGraphMouse<Integer, String> gm = new DefaultModalGraphMouse<Integer, String>();		
		gm.setMode(ModalGraphMouse.Mode.PICKING);	
		vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint); //fill color
		vv.getRenderContext().setEdgeDrawPaintTransformer(edgePaint);
		vv.getRenderContext().setVertexDrawPaintTransformer(strokecolor);
		vv.getRenderContext().setVertexStrokeTransformer(strokePaint);
		vv.getRenderContext().setVertexLabelTransformer(new ToStringLabeller<String>()); 
		vv.getRenderContext().setVertexShapeTransformer(vstr); //shape
		vv.getRenderer().getVertexLabelRenderer().setPosition(Position.CNTR);
		vv.setGraphMouse(gm);
		vv.setBackground(Color.WHITE);
		
		//设置对话框属性
		frame = new JFrame("Network diagram");
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setBounds(10, 10, 800, 700);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		frame.setContentPane(contentPane);
		
		Container c = frame.getContentPane();
		GroupLayout mainLayout = new GroupLayout(c);	
		
		scrollPane = new JScrollPane(vv);
		scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
		scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_NEVER);
		c.add(scrollPane);
		
		JButton filter = new JButton();
		filter.setText("Key nodes");
		c.add(filter);
		JButton savemap = new JButton();
		savemap.setText("Save Image");
		c.add(savemap);
		JButton savenetwork = new JButton();
		savenetwork.setText("Save Network");
		c.add(savenetwork);
		JButton restore = new JButton();
		restore.setText("Restore");
		c.add(restore);
		
		filter.addActionListener(e -> {
			for(String str : initialgraph.getVertices()) graph.removeVertex(str);
			for(String str : initialgraph.getEdges()) graph.removeEdge(str);
			for(int i=0;i<genes.size();i++) {
				String gene = genes.get(i);
				String type = types.get(i);
				if(tftg.keySet().contains(type)) {
					graph.addVertex(type);
					graph.addVertex(gene);
					graph.addEdge(type+"\t"+gene, type, gene, EdgeType.DIRECTED);
					HashMap<String, Set<String>> neighbor = survivalnetwork.get(gene);
					if(neighbor!=null && neighbor.size()>0) {
						Set<String> targets = neighbor.get(type);
						if(targets!=null && targets.size()>0) {
							for(String aa:targets) {
								graph.addVertex(type);
								graph.addVertex(aa);
								graph.addEdge(type+"\t"+aa, type, aa, EdgeType.DIRECTED);
							}
						}
					}
				} else {
					HashMap<String, Set<String>> neighbor = survivalnetwork.get(gene);
					if(neighbor!=null && neighbor.size()>0) {
						Set<String> targets = neighbor.get(type);
						if(targets!=null && targets.size()>0) {
							for(String aa:targets) {
								graph.addVertex(gene);
								graph.addVertex(aa);
								graph.addEdge(gene+"\t"+aa, gene, aa, EdgeType.UNDIRECTED);
							}
						} else {
							graph.addVertex(gene);
						}
					} else {
						graph.addVertex(gene);
					}
				}
			}
			vv.getRenderContext().setVertexFillPaintTransformer(vertexPaint);
			vv.getRenderContext().setEdgeDrawPaintTransformer(edgePaint);
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

		GroupLayout.SequentialGroup  hParalGroup01 = mainLayout.createSequentialGroup().addComponent(filter).addGap(10).addComponent(savemap).addGap(10).addComponent(savenetwork).addGap(10).addComponent(restore);
		GroupLayout.ParallelGroup hParalGroup03 = mainLayout.createParallelGroup(Alignment.TRAILING).addComponent(scrollPane, GroupLayout.DEFAULT_SIZE, 646, Short.MAX_VALUE)
				.addGroup(hParalGroup01);
		GroupLayout.SequentialGroup  hParalGroup = mainLayout.createSequentialGroup().addContainerGap().addGroup(hParalGroup03).addContainerGap();
		mainLayout.setHorizontalGroup(hParalGroup);
		
		GroupLayout.ParallelGroup hParalGroup05 = mainLayout.createParallelGroup().addComponent(filter).addComponent(savemap).addComponent(savenetwork).addComponent(restore);
		GroupLayout.SequentialGroup hParalGroup07 = mainLayout.createSequentialGroup().addContainerGap().addComponent(scrollPane, GroupLayout.DEFAULT_SIZE, 646, Short.MAX_VALUE)
				.addGap(10).addGroup(hParalGroup05).addGap(10);
		GroupLayout.ParallelGroup  hParalGroup2 = mainLayout.createParallelGroup().addGroup(hParalGroup07);
		mainLayout.setVerticalGroup(hParalGroup2);

		contentPane.setLayout(mainLayout);
		frame.setVisible(true);
	}

}
