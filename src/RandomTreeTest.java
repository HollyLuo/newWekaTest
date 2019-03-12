
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import java.awt.BorderLayout;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.*;
public class RandomTreeTest {
	//https://www.programcreek.com/java-api-examples/index.php?api=weka.classifiers.trees.RandomTree
	public static void main(String[] args) throws Exception {
	    final Instances trainingSet = DataSource.read("/Users/ling/Documents/Eclipseworkspace/Weka/NewPattern/src/test/hasValueLogic.csv");
	    if (trainingSet.classIndex() == -1)
	    	trainingSet.setClassIndex(trainingSet.numAttributes() - 1);
	    
	    final RandomTree tree = new RandomTree();
	    tree.buildClassifier(trainingSet);
	    System.out.print("train end");
	    
	    final Evaluation eval = new Evaluation(trainingSet);
	    eval.evaluateModel(tree, trainingSet);
	    
		System.out.print("test end");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
		System.out.println(tree.toString());
		visualize(tree);
	}
	
	public static void visualize(RandomTree tree) throws Exception{
		// display classifier
	     final javax.swing.JFrame jf = 
	       new javax.swing.JFrame("Weka Classifier Tree Visualizer: RandomTree");
	     jf.setSize(800,400);
	     jf.getContentPane().setLayout(new BorderLayout());
	     TreeVisualizer tv = new TreeVisualizer(null,((RandomTree)tree).graph(),new PlaceNode2());
	     jf.getContentPane().add(tv, BorderLayout.CENTER);
	     jf.addWindowListener(new java.awt.event.WindowAdapter() {
	       public void windowClosing(java.awt.event.WindowEvent e) {
	         jf.dispose();
	       }
	     });
	 
	     jf.setVisible(true);
	     tv.fitToScreen();
	}
}
