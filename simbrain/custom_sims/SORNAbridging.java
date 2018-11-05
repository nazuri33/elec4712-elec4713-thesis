package org.simbrain.custom_sims.simulations.sorn_abridging;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.*;
import java.io.File;

import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.JButton;
import javax.swing.JToggleButton;
import java.awt.geom.Point2D;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;


import org.simbrain.custom_sims.RegisteredSimulation;
import org.simbrain.custom_sims.helper_classes.ControlPanel;
import org.simbrain.custom_sims.helper_classes.NetBuilder;
import org.simbrain.custom_sims.helper_classes.OdorWorldBuilder;
import org.simbrain.custom_sims.helper_classes.PlotBuilder;
// import org.simbrain.custom_sims.helper_classes.AddSTDPRule;
// import org.simbrain.custom_sims.helper_classes.SORNNeuronRule;

import org.simbrain.network.NetworkComponent;
import org.simbrain.network.core.Network;
import org.simbrain.network.core.Neuron;
import org.simbrain.network.connections.*;
import org.simbrain.network.core.*;
import org.simbrain.network.core.NeuronUpdateRule.InputType;
import org.simbrain.network.neuron_update_rules.interfaces.BiasedUpdateRule;
import org.simbrain.network.groups.*;
import org.simbrain.network.layouts.*;
import org.simbrain.network.desktop.*;
import org.simbrain.network.neuron_update_rules.*;
import org.simbrain.network.synapse_update_rules.*;
import org.simbrain.network.synapse_update_rules.spikeresponders.*;
import org.simbrain.network.update_actions.*;
import org.simbrain.util.randomizer.*;
import org.simbrain.util.*;
import org.simbrain.workspace.*;
import org.simbrain.util.SimbrainConstants.Polarity;
import javax.swing.JInternalFrame;
import javax.swing.JComboBox;
import org.simbrain.util.math.*;
import org.simbrain.workspace.*;
import org.simbrain.workspace.gui.SimbrainDesktop;
import org.simbrain.world.odorworld.*;
import org.simbrain.world.odorworld.entities.*;
import org.simbrain.util.environment.*;
import org.simbrain.world.odorworld.entities.RotatingEntity;
import org.simbrain.world.odorworld.entities.OdorWorldEntity;
import org.simbrain.world.odorworld.sensors.TileSensor;
import org.simbrain.workspace.Coupling;
import org.simbrain.workspace.MismatchedAttributesException;
import org.simbrain.workspace.PotentialConsumer;
import org.simbrain.workspace.PotentialProducer;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import org.simbrain.world.odorworld.sensors.SmellSensor;

public class SORNAbridging extends RegisteredSimulation {

  public enum Stimulus {ABR, SCR}

  /** Network variables **/
  boolean homeoPlastic = true;
  boolean frozenWeights = false;
  int NUM_NEURONS = 200; // originally 2500 w/ "1024" comment
  int NUM_INPUTS = 9; // originally 200
  int GRID_SPACE = 25;
  int RADIUS = 400; // 100
  int EE_KIN = 10; // originally 25 (this is lambda_w?)
  int IE_KIN = (int)(0.2 * NUM_NEURONS/10);
  // int EI_KIN = NUM_NEURONS/50;
  int EI_KIN = 50;
  double TE_max = 0.75; // originally 0.5;
  double TI_max = 1.4; // originally 0.8
  double n_STDP = 0.001; // STDP learning Rate
  double time_step = 1.0; // 41.38; // each update in network advances time by 'time_step' (ms)
  double etaIP = 0.001;
  int y_loc, x_loc;

  /** Stimulus variables **/
  public Stimulus currentStim = Stimulus.ABR;
  public static boolean abridged = false;
  public static OdorWorld motionWorld;
  public static NeuronGroup motionLogic, skinReceptors, inputNeurons;
  public static PotentialProducer receptorProducers;
  public static PotentialConsumer inputConsumers;
  public static List<PotentialProducer> producers = new ArrayList<PotentialProducer>();
  public static List<PotentialConsumer> consumers = new ArrayList<PotentialConsumer>();
  public static OdorWorldEntity abridgingEntity;
  public static SmellSource abridgingTrigger;
  // String bouncerDispersion = "20.00";
  // double bouncerDispDouble = 20.00;
  double bouncerPeak = 15.0;
  double motionSpeed = 2.0;
  double smellCentreLength = 25.0;
  double numerator = 0.0;
  double denominator = Math.log10(2.0);
  NumberFormat doubleFormatter = new DecimalFormat("#0.00");
  OdorWorldEntity mouse;
  SmellSensor centreSmell;
  OdorWorldEntity leftCandle;
  OdorWorldEntity rightCandle;
  SmellSource leftBouncer;
  SmellSource rightBouncer;
  Neuron speedNeuron;

  /** GUI variables **/
  ControlPanel controlPanel;
  JInternalFrame internalFrame = new JInternalFrame("Control panel", true, true);
  LabelledItemPanel panel = new LabelledItemPanel();
  JToggleButton freezeNet = new JToggleButton("Freeze", false);
  JButton unfreezeNet = new JButton("Unfreeze");
  JToggleButton homeostaticPlasticity = new JToggleButton("Toggle", true);
  JButton abridging = new JButton("Toggle");
  JTextField stimulusSpeed = new JTextField("2");
  JTextField abrSpeedField = new JTextField("2");
  String[] stimulusType = {"abridging", "scrambling"};
  JComboBox<String> stimulusSelect = new JComboBox<String>(stimulusType);

  /** Neural net variables **/
  public static Network SORNNetwork;
  public static Network stimulusNetwork;
  public static NetworkComponent SORNComponent;
  public static NetworkComponent stimulusComponent;
  public static ArrayList<Neuron> neurons = new ArrayList<Neuron>(); // excitatory neurons
  public static ArrayList<Neuron> inhibitoryNeurons = new ArrayList<Neuron>(); // inhibitory neurons
  public static ArrayList<Neuron> inNeurons = new ArrayList<Neuron>(); // input neurons
  public static SORNNeuronRule sornRule = new SORNNeuronRule(); // excitatory neuron rule
  public static SORNNeuronRule str = new SORNNeuronRule(); // inhibitory neuron rule
  public static AddSTDPRule stdp = new AddSTDPRule();
  public static NeuronGroup ng, ngIn, input; // excitatory/inhibitory/input neuron groups
  public static PolarizedRandomizer exRand, inRand;
  public static RadialSimpleConstrainedKIn ee_con, ie_con, ei_con;
  public static SynapseGroup sg_ee, sg_ie, sg_ei, input_ee, ee_input, input_ie, ie_input;
  public static Sparse input_ee_con, ee_input_con, input_ie_con, ie_input_con;


  /**
  * Construct the simulation.
  *
  * @param desktop
  */
  public SORNAbridging(SimbrainDesktop desktop) {
    super(desktop);
  }

  public SORNAbridging() {
    super();
  }

  /**
  * Run the simulation
  */
  @Override
  public void run() {
    System.out.println("**** TEST *****");
    String FS = System.getProperty("file.separator");
    sim.getWorkspace().clearWorkspace();
    sim.getWorkspace().openWorkspace(new File("scripts" + FS + "scriptmenu" + FS + "SORNabridgingscrambling.zip"));
    // System.out.println(System.getProperty("user.dir"));

    setUpSORNNetwork();
    setUpSORNNeurons();
    setUpSORNSynapses();
    SORNNetworkInitialisation();
    setupStimulusNetwork();
    // setupGUI();

    setupListeners();
    makeControlPanel();
  }


  @Override
  public String getName() {
    return "SORN-Abridging";
  }

  @Override
  public SORNAbridging instantiate(SimbrainDesktop desktop) {
    return new SORNAbridging(desktop);
  }

  private void setUpSORNNetwork() {
    SORNComponent = new NetworkComponent("SORN Network");
    sim.getWorkspace().addWorkspaceComponent(SORNComponent);
    SORNNetwork = SORNComponent.getNetwork();
    SORNNetwork.setTimeStep(time_step);
    motionWorld = ((OdorWorldComponent) sim.getWorkspace().getComponent("oneObject.xml")).getWorld();
    mouse = (OdorWorldEntity) motionWorld.getEntity("Entity_8");
    centreSmell = (SmellSensor) mouse.getSensor("Smell-Center");
    leftCandle = (OdorWorldEntity) motionWorld.getEntity("Entity_9");
    rightCandle = (OdorWorldEntity) motionWorld.getEntity("Entity_4");
    leftBouncer = leftCandle.getSmellSource();
    rightBouncer = rightCandle.getSmellSource();
  }

  private void setUpSORNNeurons() {
    /* EXCITATORY */
    sornRule.sethIP((2*NUM_INPUTS)/NUM_NEURONS);
    sornRule.setMaxThreshold(TE_max);
    sornRule.setRefractoryPeriod(1);
    sornRule.setAddNoise(true);
    for (int i = 0; i < NUM_NEURONS; i++) {
      Neuron n = new Neuron(SORNNetwork);
      sornRule.setThreshold(TE_max * Math.random() + 0.01);
      n.setPolarity(Polarity.EXCITATORY);
      n.setUpdateRule(sornRule.deepCopy());
      neurons.add(n);
    }

    /* INHIBITORY */
    str.setEtaIP(etaIP);
    str.setRefractoryPeriod(1);
    str.setAddNoise(true);
    for (int i = 0; i < (int) (NUM_NEURONS * 0.2); i++) {
      Neuron n = new Neuron(SORNNetwork);
      str.setThreshold(TI_max * Math.random() + 0.01);
      n.setPolarity(Polarity.INHIBITORY);
      n.setUpdateRule(str.deepCopy());
      inhibitoryNeurons.add(n);
    }

    ng = new NeuronGroup(SORNNetwork, neurons);
    ngIn = new NeuronGroup(SORNNetwork, inhibitoryNeurons);
    GridLayout layout = new GridLayout(GRID_SPACE, GRID_SPACE, (int) Math.sqrt(NUM_NEURONS));
    ng.setLabel("Excitatory population");
    SORNNetwork.addGroup(ng);
    ng.setLayout(layout);
    ng.applyLayout(new Point(10,10));

    layout = new GridLayout(GRID_SPACE, GRID_SPACE, (int) Math.sqrt(0.2 * NUM_NEURONS));
    ngIn.setLabel("Inhibitory population");
    SORNNetwork.addGroup(ngIn);
    ngIn.setLayout(layout);
    x_loc = (int) (Math.sqrt(NUM_NEURONS) * GRID_SPACE + 300);
    ngIn.applyLayout(new Point(x_loc, 10));

    exRand = new PolarizedRandomizer(Polarity.EXCITATORY, ProbDistribution.UNIFORM);
    inRand = new PolarizedRandomizer(Polarity.INHIBITORY, ProbDistribution.UNIFORM);
    exRand.setParam1(0);
    exRand.setParam2(1);
    inRand.setParam1(0);
    inRand.setParam2(1);

    for (Neuron n : neurons) {
      ((SORNNeuronRule) n.getUpdateRule()).init(n);
    }

    for (int i = 0; i < NUM_INPUTS; i++) {
      Neuron n = new Neuron(SORNNetwork);
      // SpikingThresholdRule inRule = new SpikingThresholdRule();
      // inRule.setThreshold(0.96);
      sornRule.setThreshold(TE_max * Math.random() + 0.01);
      n.setPolarity(Polarity.EXCITATORY);
      n.setUpdateRule(sornRule.deepCopy());
      inNeurons.add(n);
    }

    input = new NeuronGroup(SORNNetwork, inNeurons);
    LineLayout line_layout = new LineLayout();
    input.setLabel("Input");
    SORNNetwork.addGroup(input);
    input.setLayout(line_layout);

    y_loc = (int) (Math.sqrt(NUM_INPUTS) * GRID_SPACE + 200);
    input.applyLayout(new Point(x_loc, y_loc));


  }

  private void setUpSORNSynapses() {
    ee_con = new RadialSimpleConstrainedKIn(EE_KIN, RADIUS);
    ie_con = new RadialSimpleConstrainedKIn(IE_KIN, 100000);
    ei_con = new RadialSimpleConstrainedKIn(EI_KIN, 100000);
    input_ee_con = new Sparse(0.5, false, false); // first argument (density) was 0.05 in original SORN.bsh
    ee_input_con = new Sparse(0.15, false, false); // density was originally 0.01
    input_ie_con = new Sparse(0.15, true, false); // density was originally 0.01
    ie_input_con = new Sparse(0.15, true, false); // density was originally 0.01 (needed to make this connection and I think input->ie denser, in particular, because there's so few of them. Modified the other two as well to maintain some degree of common proportionality with original densities)

    sg_ee = SynapseGroup.createSynapseGroup(ng, ng, ee_con, 1.0, exRand, inRand);
    sg_ee.setLabel("EE Synapses");
    sg_ee.setSpikeResponder(new Step(), Polarity.BOTH);
    sg_ie = SynapseGroup.createSynapseGroup(ngIn, ng, ie_con, 1.0, exRand, inRand);
    sg_ie.setLabel("IE Synapses");
    sg_ie.setSpikeResponder(new Step(), Polarity.BOTH);
    sg_ei = SynapseGroup.createSynapseGroup(ng, ngIn, ei_con, 1.0, exRand, inRand);
    sg_ei.setLabel("EI Synapses");
    sg_ei.setSpikeResponder(new Step(), Polarity.BOTH);
    SORNNetwork.addGroup(sg_ee);
    SORNNetwork.addGroup(sg_ie);
    SORNNetwork.addGroup(sg_ei);

    stdp.setLearningRate(n_STDP);
    sg_ee.setLearningRule(stdp, Polarity.BOTH);

    input_ee = SynapseGroup.createSynapseGroup(input, ng, input_ee_con, 1.0, exRand, inRand);
    ee_input = SynapseGroup.createSynapseGroup(ng, input, ee_input_con, 1.0, exRand, inRand);
    input_ie = SynapseGroup.createSynapseGroup(input, ngIn, input_ie_con, 1.0, exRand, inRand);
    ie_input = SynapseGroup.createSynapseGroup(ngIn, input, input_ie_con, 1.0, exRand, inRand);
    input_ee.setLabel("Input -> Excitatory");
    input_ee.setLearningRule(stdp, Polarity.BOTH);
    input_ee.setSpikeResponder(new Step(), Polarity.BOTH);
    ee_input.setLabel("Excitatory -> Input");
    ee_input.setLearningRule(stdp, Polarity.BOTH);
    ee_input.setSpikeResponder(new Step(), Polarity.BOTH);
    input_ie.setLabel("Input -> Inhibitory");
    input_ie.setSpikeResponder(new Step(), Polarity.BOTH);
    ie_input.setLabel("Inhibitory -> Input");
    ie_input.setSpikeResponder(new Step(), Polarity.BOTH);
    SORNNetwork.addGroup(input_ee);
    SORNNetwork.addGroup(ee_input);
    SORNNetwork.addGroup(input_ie);
    SORNNetwork.addGroup(ie_input);

  }

  private void SORNNetworkInitialisation() {
    for (Neuron n : neurons) {
      n.normalizeInhibitoryFanIn();
    }

    for (Neuron n : neurons) {
      n.normalizeExcitatoryFanIn();
    }

    for (Neuron n : input.getNeuronList()) {
      n.normalizeInhibitoryFanIn();
      n.normalizeExcitatoryFanIn();
    }
  }

  private void setupStimulusNetwork() {
    abridgingEntity = (OdorWorldEntity) motionWorld.getEntity("Entity_15");
    abridgingTrigger = abridgingEntity.getSmellSource();
    stimulusComponent = (NetworkComponent) sim.getWorkspace().getComponent("Network2");
    stimulusNetwork = stimulusComponent.getNetwork();
    speedNeuron = stimulusNetwork.getNeuronByLabel("Speed");
    // stimulusComponent.setGuiOn(false);
    stimulusNetwork.setTimeStep(time_step);
    // motionLogic = stimulusNetwork.getGroupByLabel("Motion logic");
    skinReceptors = (NeuronGroup) stimulusNetwork.getGroupByLabel("Receptors (abridging)");
    // inputNeurons = SORNNetwork.getGroupByLabel("Input");
    receptorProducers = NetworkComponent.getNeuronGroupProducer(stimulusComponent, skinReceptors, "getExternalActivations");
    inputConsumers = NetworkComponent.getNeuronGroupConsumer(SORNComponent, input, "setInputValues");
    producers.add(receptorProducers);
    consumers.add(inputConsumers);
    try {
      sim.getWorkspace().coupleOneToOne(producers, consumers);
    } catch (MismatchedAttributesException e) {
      System.out.println("MismatchedAttributesException");
    }
  }

  // private void setupGUI() {
  //   freezeNet.addActionListener(new ActionListener() {
  //     public void actionPerformed(ActionEvent arg0) {
  //       System.out.println("Weights should be frozen!");
  //       // motionNet.freezeSynapses(true);
  //       SORNNetwork.freezeSynapses(true);
  //     }
  //   });
  //   panel.addItem("Pause SORN", freezeNet);
  //
  //   unfreezeNet.addActionListener(new ActionListener() {
  //     public void actionPerformed(ActionEvent arg0) {
  //       System.out.println("Weights shouldn't be frozen any more!!!");
  //       // motionNet.freezeSynapses(false);
  //       SORNNetwork.freezeSynapses(false);
  //     }
  //   });
  //   panel.addItem("Resume SORN", unfreezeNet);
  //
  //   homeostaticPlasticity.addActionListener(new ActionListener() {
  //     public void actionPerformed(ActionEvent arg0) {
  //       if (homeoPlastic == true) {
  //         homeoPlastic = false;
  //         str.setEtaIP(0);
  //         System.out.println("Turned homeostatic plasticity off");
  //       } else if (homeoPlastic == false) {
  //         homeoPlastic = true;
  //         str.setEtaIP(etaIP);
  //         System.out.println("Turned homeostatic plasticity on");
  //       }
  //     }
  //   });
  //   panel.addItem("Homeostatic plasticity", homeostaticPlasticity);
  //
  //   abridging.addActionListener(new ActionListener() {
  //     public void actionPerformed(ActionEvent arg0) {
  //       if (abridged == true) {
  //         abridged = false;
  //         abridgingTrigger.setStimulusS("0,0");
  //         // bouncerDispDouble = bouncerDispDouble/2.0;
  //         System.out.println("Turned abridging off");
  //       } else if (abridged == false) {
  //         abridged = true;
  //         abridgingTrigger.setStimulusS("1,1");
  //         // bouncerDispDouble = 2.0*bouncerDispDouble;
  //         System.out.println("Turned abridging on");
  //       }
  //       // leftBouncer.setDispersion(bouncerDispDouble);
  //       // rightBouncer.setDispersion(bouncerDispDouble);
  //     }
  //   });
  //   panel.addItem("Abridging", abridging);
  //
  //
  //   panel.addItem("Brushing speed", stimulusSpeed);
  //
  //   JButton setSpeed = new JButton("OK");
  //   setSpeed.addActionListener(new ActionListener() {
  //     public void actionPerformed(ActionEvent arg0) {
  //       motionSpeed = Double.parseDouble(stimulusSpeed.getText());
  //       if (motionSpeed > 2.0) {
  //         smellCentreLength = 25.0 + (motionSpeed);
  //         bouncerPeak = (abridged ? 2:1)*smellCentreLength - 5.0;
  //       } else {
  //         smellCentreLength = 25.0;
  //         bouncerPeak = (abridged ? 2:1)*15.0;
  //       }
  //
  //       centreSmell.setRadius(smellCentreLength);
  //       ((BiasedUpdateRule) speedNeuron.getUpdateRule()).setBias(motionSpeed);
  //       // we want to set the dispersion of the candles to something like 10*log2(speed)
  //       // Java only has logE, log10, so we'll have to do change of base (i.e. 10*(log10(speed)/log10(2)))
  //       // numerator = Math.log10(motionSpeed);
  //
  //       // bouncerDispersion = doubleFormatter.format(10*(numerator/denominator));
  //       // bouncerDispDouble = (abridged ? 2 : 1)*Double.parseDouble(bouncerDispersion);
  //       // leftBouncer.setDispersion(bouncerDispDouble);
  //       // rightBouncer.setDispersion(bouncerDispDouble);
  //       leftBouncer.setDispersion(bouncerPeak + 5.0);
  //       rightBouncer.setDispersion(bouncerPeak + 5.0);
  //       leftBouncer.setPeak(bouncerPeak);
  //       rightBouncer.setPeak(bouncerPeak);
  //       sim.getWorkspace().iterate();
  //     }
  //   });
  //   panel.setMyNextItemRow(4);
  //   panel.addItem("", setSpeed, 1);
  //
  //   internalFrame.setLocation(0,0);
  //   internalFrame.getContentPane().add(panel);
  //   internalFrame.setVisible(true);
  //   internalFrame.pack();
  //   sim.getDesktop().addInternalFrame(internalFrame);
  //   // SORNComponent.getParentFrame().setBounds(784, 0, 583, 402);
  //   sim.getDesktop().getDesktopComponent(sim.getWorkspace().getComponent("SORN Network")).getParentFrame().setBounds(784, 1, 583, 402);
  //   sim.getDesktop().getDesktopComponent(sim.getWorkspace().getComponent("oneObject.xml")).setLocation(1371, 1);
  // }

  void setupListeners() {
    stimulusSelect.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent arg0) {
        String selectedStimulus = (String) stimulusSelect.getSelectedItem();
        if (selectedStimulus.equals("abridging")) {
          currentStim = Stimulus.ABR;
        } else {
          currentStim = Stimulus.SCR;
        }
      }
    });

    freezeNet.addChangeListener(new ChangeListener() {
      public void stateChanged(ChangeEvent ch) {
        frozenWeights = !frozenWeights;
        SORNNetwork.freezeSynapses(frozenWeights);
      }
    });

    homeostaticPlasticity.addChangeListener(new ChangeListener() {
      public void stateChanged(ChangeEvent ch) {
        homeoPlastic = !homeoPlastic;
        double eta_ip = homeoPlastic ? etaIP : 0;
        str.setEtaIP(eta_ip);
      }
    });

  }

  void makeControlPanel() {
    controlPanel = ControlPanel.makePanel(sim, "Control panel", -6, 1);
    controlPanel.addItem("Stimulus type", stimulusSelect);
    controlPanel.addItem("Freeze/unfreeze weights", freezeNet);
    controlPanel.addItem("Homeostatic plasticity", homeostaticPlasticity);
    controlPanel.addSeparator();

    controlPanel.addButton("Toggle", "Abridging", () -> {
      abridged = !abridged;
      String stimVector = abridged ? "1,1" : "0,0";
      abridgingTrigger.setStimulusS(stimVector);
    });

    abrSpeedField = controlPanel.addTextField("Brushing speed", "" + motionSpeed);
    controlPanel.addButton("Stop", () -> {
      sim.getWorkspace().stop();
    });

  }

}
