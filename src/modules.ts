import * as PIXI from "pixi.js";
import { NodeModule, Nodes } from "./nodes";
import { Vector2, getMapVectorFromScreen } from "./math";
import { Button, SelectButton } from "./button";
import * as tf from "@tensorflow/tfjs";

class Module extends PIXI.Container {
  nodeModules: NodeModule[];
  dragging: boolean;
  static currentZindex: number = 0;

  constructor() {
    super();
    this.nodeModules = [];
    this.eventMode = "static";
    this.dragging = false;
  }
  update(nodes: Nodes): void {
    this.nodeModules.forEach((n) =>
      n.setGlobalVector(new Vector2(this.x, this.y))
    );
    nodes.update();
  }
  registerEvents(
    app: PIXI.Application<PIXI.ICanvas>,
    view: PIXI.Container<PIXI.DisplayObject>,
    nodes: Nodes
  ) {
    const dragOffset = new Vector2(0, 0);

    this.on("pointerdown", (event) => {
      if (event.button != 0) return;

      this.dragging = true;

      const vector = getMapVectorFromScreen(
        new Vector2(event.x, event.y),
        view
      );
      dragOffset.set(vector.sub(new Vector2(this.x, this.y)));
      this.zIndex = ++Module.currentZindex;

      app.stage.on("pointermove", onPointerMove);
      app.stage.on("pointerup", onPointerUp);
      app.stage.on("pointerleave", onPointerUp);
    });
    const onPointerMove = (event: PIXI.FederatedPointerEvent) => {
      if (!this.dragging) return;

      const vector = getMapVectorFromScreen(
        new Vector2(event.x, event.y),
        view
      );

      this.x = vector.x - dragOffset.x;
      this.y = vector.y - dragOffset.y;

      this.update(nodes);
    };
    const onPointerUp = () => {
      this.dragging = false;

      app.stage.off("pointermove", onPointerMove);
      app.stage.off("pointerup", onPointerUp);
      app.stage.off("pointerleave", onPointerUp);
    };
  }
}

class CoreModule extends Module {
  text: PIXI.Text;
  graphic: PIXI.Graphics;
  inputModule: NodeModule;
  inputCopyModule: NodeModule;
  outputModule: NodeModule;

  trainMode: boolean;
  trainModeButton: PIXI.Graphics;
  trainModeText: PIXI.Text;
  trainButton: Button;

  xs: tf.Tensor[];
  ys: tf.Tensor[];

  model: tf.Sequential;
  trainCompleted: boolean;
  trainning: boolean;
  inputShape: number[];
  outputShape: number;

  epochs: number;

  onModuleEditOpen: () => void;

  constructor(nodes: Nodes) {
    super();

    this.onModuleEditOpen = () => {};

    this.epochs = 500;

    this.inputShape = [1];
    this.outputShape = 1;
    this.model = tf.sequential();
    // this.model.add(
    //   tf.layers.conv2d({
    //     inputShape: this.inputShape,
    //     kernelSize: 5,
    //     filters: 8,
    //     strides: 1,
    //     activation: "relu",
    //     kernelInitializer: "varianceScaling",
    //   })
    // );
    // this.model.add(
    //   tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
    // );

    // this.model.add(
    //   tf.layers.conv2d({
    //     kernelSize: 5,
    //     filters: 16,
    //     strides: 1,
    //     activation: "relu",
    //     kernelInitializer: "varianceScaling",
    //   })
    // );
    // this.model.add(
    //   tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
    // );
    // this.model.add(tf.layers.flatten());
    // this.model.add(
    //   tf.layers.dense({
    //     units: this.outputShape,
    //     kernelInitializer: "varianceScaling",
    //     activation: "softmax",
    //   })
    // );
    // this.model.compile({
    //   optimizer: tf.train.adam(),
    //   loss: "categoricalCrossentropy",
    //   metrics: ["accuracy"],
    // });

    // this.inputShape = [2];
    // this.inputShape = [2];
    // this.outputShape = 1;
    // this.model = tf.sequential();
    // this.model.add(
    //   tf.layers.dense({
    //     units: 64,
    //     inputShape: this.inputShape,
    //     activation: "relu",
    //   })
    // );
    // this.model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    // this.model.add(tf.layers.dense({ units: this.outputShape }));

    // this.model.compile({
    //   optimizer: tf.train.adam(),
    //   loss: "meanSquaredError",
    // });

    this.xs = [];
    this.ys = [];

    this.trainCompleted = false;

    this.graphic = new PIXI.Graphics();
    this.graphic.beginFill(0x222222);
    this.graphic.drawRect(0, 0, 200, 200);
    this.graphic.endFill();

    this.trainMode = false;
    this.trainModeButton = new PIXI.Graphics();
    this.trainModeText = new PIXI.Text("Train mode", {
      fill: 0xffffff,
      align: "center",
      fontSize: 15,
    });
    this.trainButton = new Button(
      new Vector2(100, 50),
      0x333333,
      0xffffff,
      "Start Train",
      20
    );

    this.trainModeButton.on("pointerdown", (event) => event.stopPropagation());

    this.trainButton.x = 100;
    this.trainButton.y = 150;

    this.text = new PIXI.Text("Core", {
      wordWrapWidth: 200,
      wordWrap: true,
      fill: 0xffffff,
      align: "center",
      fontSize: 35,
    });
    this.text.anchor.set(0.5, 0.5);
    this.text.x = 200 / 2;
    this.text.y = 200 / 2;

    this.addChild(this.graphic);
    this.addChild(this.text);

    this.inputModule = new NodeModule(nodes, "Input");
    this.inputCopyModule = new NodeModule(nodes, "Input Copy");
    this.outputModule = new NodeModule(nodes, "Output");

    this.inputModule.x = 0;
    this.inputModule.y = 100;
    this.inputCopyModule.x = 200;
    this.inputCopyModule.y = 50;
    this.outputModule.x = 200;
    this.outputModule.y = 100;

    this.inputModule.setGlobalVector(new Vector2(this.x, this.y));
    this.inputCopyModule.setGlobalVector(new Vector2(this.x, this.y));
    this.outputModule.setGlobalVector(new Vector2(this.x, this.y));

    this.inputModule.onDataPassed = (data, passCount) => {
      if (
        !Array.isArray(data) ||
        data.every(
          (v) =>
            !(v instanceof tf.Tensor) ||
            [...(v.shape as number[])].sort().toString() !=
              [...this.inputShape].sort().toString()
        )
      )
        return;

      if (this.trainMode) {
        this.xs.push(...(data as tf.Tensor[]));

        return;
      }

      if (!this.trainCompleted) return;

      this.inputCopyModule.passData(data, passCount);

      const result = [
        ...((
          this.model.predict(tf.stack(data as tf.Tensor[])) as tf.Tensor
        ).arraySync() as tf.TensorLike[]),
      ].map((v) => tf.tensor(v));

      this.outputModule.passData(result, passCount);
    };
    this.outputModule.onDataPassed = (data) => {
      if (!this.trainMode) return;

      if (
        !Array.isArray(data) ||
        data.every(
          (v) =>
            !(v instanceof tf.Tensor) ||
            [...(v.shape as number[])].sort().toString() !=
              [this.outputShape].toString()
        )
      )
        return;

      this.ys.push(...(data as tf.Tensor[]));
    };

    this.addChild(this.inputModule);
    this.addChild(this.inputCopyModule);
    this.addChild(this.outputModule);

    this.nodeModules.push(
      this.inputModule,
      this.outputModule,
      this.inputCopyModule
    );

    this.addChild(this.trainButton);

    const trainModeButtonDraw = () => {
      this.trainModeButton.clear();
      this.trainModeButton.beginFill(0x000000);
      this.trainModeButton.drawRoundedRect(-30, -10, 60, 20, 10);
      this.trainModeButton.endFill();

      if (this.trainMode) {
        this.trainModeButton.beginFill(0x11aa11);
        this.trainModeButton.drawCircle(20, 0, 10);
        this.trainModeButton.endFill();
      } else {
        this.trainModeButton.beginFill(0xaa1111);
        this.trainModeButton.drawCircle(-20, 0, 10);
        this.trainModeButton.endFill();
      }
    };
    trainModeButtonDraw();

    this.trainModeButton.eventMode = "static";

    this.trainModeButton.on("click", () => {
      this.trainMode = !this.trainMode;
      trainModeButtonDraw();
    });

    this.trainning = false;
    this.trainButton.on("click", () => {
      if (this.xs.length == 0 || this.ys.length == 0) return;

      if (this.trainning) return;
      this.trainning = true;

      this.model
        .fit(tf.stack(this.xs), tf.stack(this.ys), {
          epochs: this.epochs,
          batchSize: 24,
          validationSplit: 0.2,
          callbacks: {
            onEpochEnd: (epoch, logs) => {
              console.log(
                `Epoch ${epoch}` + (logs?.loss ? `: loss=${logs?.loss}` : ``)
              );
              this.trainButton.text.text = `${(
                (epoch / this.epochs) *
                100
              ).toFixed(2)}%`;
            },
          },
        })
        .then(() => {
          this.trainning = false;
          this.trainCompleted = true;
          this.trainButton.text.text = "Start Train";
        });
    });

    let lastClick: number | null = null;
    this.on("click", () => {
      if (lastClick != null && Date.now() - lastClick < 200) {
        this.onModuleEditOpen();
      } else {
        lastClick = Date.now();
      }
    });

    this.trainModeButton.x = 50;
    this.trainModeButton.y = 180;

    this.addChild(this.trainModeButton);

    this.trainModeText.anchor.set(0.5, 0.5);
    this.trainModeText.x = 50;
    this.trainModeText.y = 160;
    this.addChild(this.trainModeText);
  }
}
class PointInputModule extends Module {
  text: PIXI.Text;
  graphic: PIXI.Graphics;
  bodyContainer: PIXI.Container;
  background: PIXI.Graphics;
  outputXModule: NodeModule;
  outputYModule: NodeModule;
  zoominButton: Button;
  zoomoutButton: Button;
  resetButton: Button;
  passButton: Button;
  zoomSize: number;
  points: Vector2[];
  backgroundUpdate: () => void;

  constructor(nodes: Nodes, view: PIXI.Container) {
    super();

    this.points = [];

    this.graphic = new PIXI.Graphics();
    this.graphic.beginFill(0x222222);
    this.graphic.drawRect(0, 0, 500, 500);
    this.graphic.endFill();

    this.text = new PIXI.Text("Point Input", {
      fill: 0xffffff,
      fontSize: 35,
    });
    this.text.x = 20;
    this.text.y = 20;

    this.addChild(this.graphic);
    this.addChild(this.text);

    this.bodyContainer = new PIXI.Container();
    this.bodyContainer.x = 20;
    this.bodyContainer.y = 80;

    const width = 500 - 40;
    const height = 500 - 120;

    const mask = new PIXI.Graphics();
    mask.beginFill(0xffffff);
    mask.drawRect(0, 0, width, height);
    mask.endFill();
    this.bodyContainer.addChild(mask);
    this.bodyContainer.mask = mask;

    this.addChild(this.bodyContainer);

    this.background = new PIXI.Graphics();

    this.bodyContainer.addChild(this.background);

    this.zoomSize = 6;

    this.backgroundUpdate = () => {
      this.background.clear();
      this.background.beginFill(0x000000);
      this.background.drawRect(0, 0, width, height);
      this.background.endFill();

      this.background.lineStyle(1, 0xffffff);
      for (let y = 0; y < height / this.zoomSize; y += 5) {
        this.background.moveTo(0, height - y * this.zoomSize);
        this.background.lineTo(width, height - y * this.zoomSize);
      }
      for (let x = 0; x < width / this.zoomSize; x += 5) {
        this.background.moveTo(x * this.zoomSize, 0);
        this.background.lineTo(x * this.zoomSize, height);
      }

      this.background.beginFill(0xffffff);
      this.points.forEach((vector) => {
        this.background.drawCircle(
          vector.x * this.zoomSize,
          height - vector.y * this.zoomSize,
          10 * (this.zoomSize / 4)
        );
      });
      this.background.endFill();
    };
    this.backgroundUpdate();

    this.bodyContainer.eventMode = "static";
    this.background.eventMode = "static";

    this.background.on("pointerdown", (event) => {
      event.stopPropagation();
    });
    this.background.on("click", (event) => {
      const vector = getMapVectorFromScreen(
        new Vector2(event.x, event.y),
        view
      ).sub(
        new Vector2(
          this.x + this.bodyContainer.x,
          this.y + this.bodyContainer.y
        )
      );
      vector.set(new Vector2(vector.x, height - vector.y).div(this.zoomSize));

      this.points.push(vector);
      this.backgroundUpdate();
    });

    this.zoominButton = new Button(
      new Vector2(30, 30),
      0x333333,
      0xffffff,
      "+",
      35
    );
    this.zoomoutButton = new Button(
      new Vector2(30, 30),
      0x333333,
      0xffffff,
      "-",
      35
    );
    this.resetButton = new Button(
      new Vector2(30, 30),
      0x333333,
      0xffffff,
      "🗑️",
      20
    );
    this.passButton = new Button(
      new Vector2(100, 50),
      0x333333,
      0xffffff,
      "Pass",
      35
    );

    this.zoominButton.x = 20;
    this.zoominButton.y = 500 - 30;
    this.zoomoutButton.x = 50;
    this.zoomoutButton.y = 500 - 30;
    this.resetButton.x = 80;
    this.resetButton.y = 500 - 30;
    this.passButton.x = 500 - 100;
    this.passButton.y = 0;

    this.addChild(this.zoominButton);
    this.addChild(this.zoominButton);
    this.addChild(this.zoomoutButton);
    this.addChild(this.resetButton);
    this.addChild(this.passButton);

    this.zoominButton.on("click", () => {
      this.zoomSize *= 1.2;
      this.backgroundUpdate();
    });
    this.zoomoutButton.on("click", () => {
      this.zoomSize *= 0.8;
      this.backgroundUpdate();
    });
    this.resetButton.on("click", () => {
      this.points = [];
      this.backgroundUpdate();
    });
    this.passButton.on("click", () => {
      this.outputXModule.passData(this.points.map((point) => point.x));
      this.outputYModule.passData(this.points.map((point) => point.y));
    });

    this.outputXModule = new NodeModule(nodes, "Output X");
    this.outputYModule = new NodeModule(nodes, "Output Y");

    this.nodeModules.push(this.outputXModule);
    this.nodeModules.push(this.outputYModule);

    this.outputXModule.x = 500 / 2;
    this.outputXModule.y = 500;
    this.outputYModule.x = 500 / 2 + 100;
    this.outputYModule.y = 500;

    this.outputXModule.setGlobalVector(new Vector2(this.x, this.y));
    this.outputYModule.setGlobalVector(new Vector2(this.x, this.y));

    this.addChild(this.outputXModule);
    this.addChild(this.outputYModule);
  }
}
class PointZInputModule extends PointInputModule {
  outputZModule: NodeModule;
  pointsZ: number[];

  constructor(nodes: Nodes, view: PIXI.Container) {
    super(nodes, view);

    this.text.text = "PointZ Input";

    this.pointsZ = [];

    this.background.off("click");

    let drawing = false;
    const startVector = new Vector2(0, 0);
    const lastVector = new Vector2(0, 0);

    this.background.on("pointerdown", (event) => {
      drawing = true;

      const vector = getMapVectorFromScreen(
        new Vector2(event.x, event.y),
        view
      ).sub(
        new Vector2(
          this.x + this.bodyContainer.x,
          this.y + this.bodyContainer.y
        )
      );
      startVector.set(
        new Vector2(vector.x, height - vector.y).div(this.zoomSize)
      );
      lastVector.set(lastVector);

      this.points.push(startVector);
      this.pointsZ.push(255);

      this.backgroundUpdate();
    });
    this.background.on("pointermove", (event) => {
      if (!drawing) return;

      const vector = getMapVectorFromScreen(
        new Vector2(event.x, event.y),
        view
      ).sub(
        new Vector2(
          this.x + this.bodyContainer.x,
          this.y + this.bodyContainer.y
        )
      );
      vector.set(new Vector2(vector.x, height - vector.y).div(this.zoomSize));

      if (vector.distance(lastVector) < 0.5 * this.zoomSize) return;

      lastVector.set(vector);

      const z = 255 - startVector.distance(vector) * this.zoomSize;

      this.points.push(vector);
      this.pointsZ.push(z < 0 ? 0 : z);

      this.backgroundUpdate();
    });
    this.background.on("pointerleave", () => {
      drawing = false;
    });
    this.background.on("pointerup", () => {
      drawing = false;
    });
    this.passButton.on("click", () => {
      this.outputZModule.passData(this.pointsZ.map((z) => z));
    });

    const width = 500 - 40;
    const height = 500 - 120;

    this.backgroundUpdate = () => {
      this.background.clear();
      this.background.beginFill(0x000000);
      this.background.drawRect(0, 0, width, height);
      this.background.endFill();

      this.background.lineStyle(1, 0xffffff);
      for (let y = 0; y < height / this.zoomSize; y += 5) {
        this.background.moveTo(0, height - y * this.zoomSize);
        this.background.lineTo(width, height - y * this.zoomSize);
      }
      for (let x = 0; x < width / this.zoomSize; x += 5) {
        this.background.moveTo(x * this.zoomSize, 0);
        this.background.lineTo(x * this.zoomSize, height);
      }
      this.background.lineStyle(0);

      this.points.forEach((vector, i) => {
        const z = this.pointsZ[i] / 255;
        this.background.beginFill(new PIXI.Color([z, z, z]));
        this.background.drawCircle(
          vector.x * this.zoomSize,
          height - vector.y * this.zoomSize,
          10 * (this.zoomSize / 4)
        );
        this.background.endFill();
      });
    };

    this.outputZModule = new NodeModule(nodes, "Output Z");
    this.outputZModule.x = 500 / 2 + 200;
    this.outputZModule.y = 500;

    this.outputZModule.setGlobalVector(new Vector2(this.x, this.y));

    this.addChild(this.outputZModule);

    this.nodeModules.push(this.outputZModule);
  }
}
class PointOutputModule extends Module {
  text: PIXI.Text;
  graphic: PIXI.Graphics;
  bodyContainer: PIXI.Container;
  background: PIXI.Graphics;
  zoominButton: Button;
  zoomoutButton: Button;
  resetButton: Button;
  zoomSize: number;
  points: Vector2[];
  inputXModule: NodeModule;
  inputYModule: NodeModule;
  pointsXQueue: number[];
  pointsYQueue: number[];
  queueInterval: NodeJS.Timeout;
  backgroundUpdate: () => void;

  constructor(nodes: Nodes) {
    super();

    this.points = [];
    this.pointsXQueue = [];
    this.pointsYQueue = [];

    this.graphic = new PIXI.Graphics();
    this.graphic.beginFill(0x222222);
    this.graphic.drawRect(0, 0, 500, 500);
    this.graphic.endFill();

    this.text = new PIXI.Text("Point Output", {
      fill: 0xffffff,
      fontSize: 35,
    });
    this.text.x = 20;
    this.text.y = 20;

    this.addChild(this.graphic);
    this.addChild(this.text);

    this.bodyContainer = new PIXI.Container();
    this.bodyContainer.x = 20;
    this.bodyContainer.y = 80;

    const width = 500 - 40;
    const height = 500 - 120;

    const mask = new PIXI.Graphics();
    mask.beginFill(0xffffff);
    mask.drawRect(0, 0, width, height);
    mask.endFill();
    this.bodyContainer.addChild(mask);
    this.bodyContainer.mask = mask;

    this.addChild(this.bodyContainer);

    this.background = new PIXI.Graphics();

    this.bodyContainer.addChild(this.background);

    this.zoomSize = 6;

    this.backgroundUpdate = () => {
      this.background.clear();
      this.background.beginFill(0x000000);
      this.background.drawRect(0, 0, width, height);
      this.background.endFill();

      this.background.lineStyle(1, 0xffffff);
      for (let y = 0; y < height / this.zoomSize; y += 5) {
        this.background.moveTo(0, height - y * this.zoomSize);
        this.background.lineTo(width, height - y * this.zoomSize);
      }
      for (let x = 0; x < width / this.zoomSize; x += 5) {
        this.background.moveTo(x * this.zoomSize, 0);
        this.background.lineTo(x * this.zoomSize, height);
      }
      this.background.lineStyle(0);

      this.background.beginFill(0xffffff);
      this.points.forEach((vector) => {
        this.background.drawCircle(
          vector.x * this.zoomSize,
          height - vector.y * this.zoomSize,
          10 * (this.zoomSize / 4)
        );
      });
      this.background.endFill();
    };
    this.backgroundUpdate();

    this.zoominButton = new Button(
      new Vector2(30, 30),
      0x333333,
      0xffffff,
      "+",
      35
    );
    this.zoomoutButton = new Button(
      new Vector2(30, 30),
      0x333333,
      0xffffff,
      "-",
      35
    );
    this.resetButton = new Button(
      new Vector2(30, 30),
      0x333333,
      0xffffff,
      "🗑️",
      20
    );

    this.zoominButton.x = 20;
    this.zoominButton.y = 500 - 30;
    this.zoomoutButton.x = 50;
    this.zoomoutButton.y = 500 - 30;
    this.resetButton.x = 80;
    this.resetButton.y = 500 - 30;

    this.addChild(this.zoominButton);
    this.addChild(this.zoominButton);
    this.addChild(this.zoomoutButton);
    this.addChild(this.resetButton);

    this.zoominButton.on("click", () => {
      this.zoomSize *= 1.2;
      this.backgroundUpdate();
    });
    this.zoomoutButton.on("click", () => {
      this.zoomSize *= 0.8;
      this.backgroundUpdate();
    });
    this.resetButton.on("click", () => {
      this.points = [];
      this.pointsXQueue = [];
      this.pointsYQueue = [];
      this.backgroundUpdate();
    });

    this.inputXModule = new NodeModule(nodes, "Input X");
    this.inputYModule = new NodeModule(nodes, "Input Y");

    this.nodeModules.push(this.inputXModule);
    this.nodeModules.push(this.inputYModule);

    this.inputXModule.x = 500 / 2;
    this.inputXModule.y = 0;
    this.inputYModule.x = 500 / 2 + 100;
    this.inputYModule.y = 0;

    this.inputXModule.setGlobalVector(new Vector2(this.x, this.y));
    this.inputYModule.setGlobalVector(new Vector2(this.x, this.y));

    this.addChild(this.inputXModule);
    this.addChild(this.inputYModule);

    this.inputXModule.onDataPassed = (data) => {
      if (
        !(Array.isArray(data) && data.every((item) => typeof item === "number"))
      )
        return;

      this.pointsXQueue.push(...data);
    };
    this.inputYModule.onDataPassed = (data) => {
      if (
        !(Array.isArray(data) && data.every((item) => typeof item === "number"))
      )
        return;

      this.pointsYQueue.push(...data);
    };

    this.queueInterval = setInterval(() => {
      if (this.pointsYQueue.length != 0 && this.pointsXQueue.length != 0) {
        this.points.push(
          new Vector2(this.pointsXQueue[0], this.pointsYQueue[0])
        );
        this.pointsXQueue.splice(0, 1);
        this.pointsYQueue.splice(0, 1);

        this.backgroundUpdate();
      }
    }, 50);
  }
}
class PointZOutputModule extends PointOutputModule {
  inputZModule: NodeModule;
  pointsZQueue: number[];
  pointsZ: number[];

  constructor(nodes: Nodes) {
    super(nodes);

    this.text.text = "PointZ Output";

    this.inputZModule = new NodeModule(nodes, "Output Z");
    this.inputZModule.x = 500 / 2 + 200;
    this.inputZModule.y = 0;

    this.pointsZ = [];

    this.inputZModule.setGlobalVector(new Vector2(this.x, this.y));

    this.addChild(this.inputZModule);

    this.nodeModules.push(this.inputZModule);

    this.pointsZQueue = [];
    this.inputZModule.onDataPassed = (data) => {
      if (
        !(Array.isArray(data) && data.every((item) => typeof item === "number"))
      )
        return;

      this.pointsZQueue.push(...data);
    };

    const width = 500 - 40;
    const height = 500 - 120;

    this.backgroundUpdate = () => {
      this.background.clear();
      this.background.beginFill(0x000000);
      this.background.drawRect(0, 0, width, height);
      this.background.endFill();

      this.points.forEach((vector, i) => {
        const z = Math.min(Math.max(this.pointsZ[i], 0), 255) / 255;

        this.background.beginFill(new PIXI.Color([z, z, z]));

        this.background.drawRect(
          vector.x * this.zoomSize,
          height - (vector.y + 1) * this.zoomSize,
          this.zoomSize * 0.5,
          this.zoomSize * 0.5
        );
        this.background.endFill();
      });
    };
    this.backgroundUpdate();

    clearInterval(this.queueInterval);
    this.queueInterval = setInterval(() => {
      for (let i = 0; i < 100; i++) {
        if (
          this.pointsYQueue.length != 0 &&
          this.pointsXQueue.length != 0 &&
          this.pointsZQueue.length != 0
        ) {
          this.points.push(
            new Vector2(this.pointsXQueue[0], this.pointsYQueue[0])
          );
          this.pointsZ.push(this.pointsZQueue[0]);

          this.pointsXQueue.splice(0, 1);
          this.pointsYQueue.splice(0, 1);
          this.pointsZQueue.splice(0, 1);
        } else {
          break;
        }
      }

      this.backgroundUpdate();
    }, 10);
  }
}
class RangeInputModule extends Module {
  text: PIXI.Text;
  rangeText: PIXI.Text;
  graphic: PIXI.Graphics;
  bodyContainer: PIXI.Container;
  outputModules: NodeModule[];
  dimension: number;
  passButton: Button;
  points: Vector2[];
  rangeStart: number;
  rangeEnd: number;
  rangeStep: number;

  constructor(nodes: Nodes, dimension: number) {
    super();

    this.points = [];
    this.rangeStart = 0;
    this.rangeEnd = 80;
    this.rangeStep = 0.5;
    this.dimension = dimension;

    const width = 300 + (dimension - 1) * 100;

    this.graphic = new PIXI.Graphics();
    this.graphic.beginFill(0x222222);
    this.graphic.drawRect(0, 0, width, 150);
    this.graphic.endFill();

    this.text = new PIXI.Text(`Range ${dimension}D`, {
      fill: 0xffffff,
      align: "center",
      fontSize: 35,
    });
    this.text.x = 20;
    this.text.y = 20;

    this.rangeText = new PIXI.Text("range(0, 80, 0.5)", {
      wordWrapWidth: 380,
      wordWrap: true,
      fill: 0xffffff,
      fontSize: 30,
    });
    this.rangeText.x = 20;
    this.rangeText.y = 80;

    this.addChild(this.graphic);
    this.addChild(this.text);
    this.addChild(this.rangeText);

    this.bodyContainer = new PIXI.Container();
    this.bodyContainer.x = 20;
    this.bodyContainer.y = 80;

    this.passButton = new Button(
      new Vector2(100, 50),
      0x333333,
      0xffffff,
      "Pass",
      35
    );

    this.passButton.x = width - 100;
    this.passButton.y = 0;

    this.addChild(this.passButton);

    this.outputModules = [];

    this.passButton.on("click", () => {
      const range = [];

      for (let i = this.rangeStart; i < this.rangeEnd; i += this.rangeStep) {
        range.push(i);
      }

      const data: number[][] = new Array(this.dimension);

      for (let i = 0; i < this.dimension; i++) {
        data[i] = new Array(Math.pow(range.length, this.dimension));
      }
      for (let i = 0; i < Math.pow(range.length, this.dimension); i++) {
        const coord = new Array(this.dimension);
        let index = i;

        for (let dim = 0; dim < this.dimension; dim++) {
          coord[dim] = index % range.length;
          index = Math.floor(index / range.length);
        }

        for (let dim = 0; dim < this.dimension; dim++) {
          data[dim][i] = range[coord[dim]];
        }
      }
      this.outputModules.forEach((outputModule, i) => {
        outputModule.passData(data[i]);
      });
    });

    for (let i = 0; i < dimension; i++) {
      const outputModule = new NodeModule(nodes, `Output ${i}`);

      this.nodeModules.push(outputModule);

      outputModule.x = 400 / 2 + 100 * i;
      outputModule.y = 150;

      outputModule.setGlobalVector(new Vector2(this.x, this.y));

      this.addChild(outputModule);
      this.outputModules.push(outputModule);
    }
  }
}
class TensorPackModule extends Module {
  text: PIXI.Text;
  graphic: PIXI.Graphics;
  inputModules: NodeModule[];
  inputQueue: Map<number, number[]>;
  outputModule: NodeModule;
  dimension: number;
  resetButton: Button;

  constructor(nodes: Nodes, dimension: number) {
    super();

    this.dimension = dimension;
    this.inputModules = [];

    this.inputQueue = new Map();

    const height = this.dimension * 50 + 140;

    this.graphic = new PIXI.Graphics();
    this.graphic.beginFill(0x222222);
    this.graphic.drawRect(0, 0, 150, height);
    this.graphic.endFill();

    this.text = new PIXI.Text("Pack", {
      fill: 0xffffff,
      align: "center",
      fontSize: 35,
    });
    this.text.x = 20;
    this.text.y = 20;

    this.resetButton = new Button(
      new Vector2(30, 30),
      0x333333,
      0xffffff,
      "🗑️",
      20
    );

    this.resetButton.x = 0;
    this.resetButton.y = height - 30;

    this.addChild(this.graphic);
    this.addChild(this.text);
    this.addChild(this.resetButton);

    this.resetButton.on("click", () => {
      this.inputQueue.clear();
    });

    for (let i = 0; i < this.dimension; i++) {
      const module = new NodeModule(nodes, `Input ${i}`);
      this.inputModules.push(module);
      module.x = 0;
      module.y = 80 + i * 50;
      this.nodeModules.push(module);
      this.addChild(module);

      module.onDataPassed = (data) => {
        if (
          !(
            Array.isArray(data) &&
            data.every((item) => typeof item === "number")
          )
        )
          return;

        if (this.inputQueue.has(i)) {
          this.inputQueue.get(i)!.push(...data);
        } else {
          this.inputQueue.set(i, data);
        }

        const inputArray: number[][] = [];

        for (let i = 0; i < this.dimension; i++) {
          if (!this.inputQueue.has(i)) break;

          inputArray.push(this.inputQueue.get(i)!);
        }

        if (inputArray.length != this.dimension) return;

        const results: tf.Tensor[] = [];

        while (inputArray.every((queue) => queue.length != 0)) {
          results.push(tf.tensor(inputArray.map((queue) => queue[0])));
          inputArray.forEach((queue) => queue.splice(0, 1));
        }
        this.outputModule.passData(results);
      };
    }

    this.outputModule = new NodeModule(nodes, "Output");

    this.nodeModules.push(this.outputModule);

    this.outputModule.x = 150;
    this.outputModule.y = height / 2;

    this.outputModule.setGlobalVector(new Vector2(this.x, this.y));

    this.addChild(this.outputModule);
  }
}
class TensorUnpackModule extends Module {
  text: PIXI.Text;
  graphic: PIXI.Graphics;
  inputModules: NodeModule[];
  inputModule: NodeModule;
  dimension: number;

  constructor(nodes: Nodes, dimension: number) {
    super();

    this.dimension = dimension;
    this.inputModules = [];

    const height = this.dimension * 50 + 140;

    this.graphic = new PIXI.Graphics();
    this.graphic.beginFill(0x222222);
    this.graphic.drawRect(0, 0, 150, height);
    this.graphic.endFill();

    this.text = new PIXI.Text("Unpack", {
      fill: 0xffffff,
      align: "center",
      fontSize: 35,
    });
    this.text.x = 20;
    this.text.y = 20;

    this.addChild(this.graphic);
    this.addChild(this.text);

    for (let i = 0; i < this.dimension; i++) {
      const module = new NodeModule(nodes, `Output ${i}`);
      this.inputModules.push(module);
      module.x = 150;
      module.y = 80 + i * 50;
      this.nodeModules.push(module);
      this.addChild(module);
    }

    this.inputModule = new NodeModule(nodes, "Input");

    this.nodeModules.push(this.inputModule);

    this.inputModule.x = 0;
    this.inputModule.y = height / 2;

    this.inputModule.setGlobalVector(new Vector2(this.x, this.y));

    this.inputModule.onDataPassed = (data) => {
      if (
        !(
          Array.isArray(data) &&
          data.length > 0 &&
          data.every(
            (arr) => arr instanceof tf.Tensor && arr.shape[0] == this.dimension
          )
        )
      )
        return;

      const tensorData = data.map(
        (tensor) => (tensor as tf.Tensor).arraySync() as number[]
      );

      const result: number[][] = [];

      for (let i = 0; i < this.dimension; i++) {
        const curr = [];

        for (let j = 0; j < tensorData.length; j++) {
          curr.push(tensorData[j][i]);
        }
        result.push(curr);
      }

      result.forEach((arr, i) => {
        this.inputModules[i].passData(arr);
      });
    };

    this.addChild(this.inputModule);
  }
}
class DrawingModule extends Module {
  text: PIXI.Text;
  graphic: PIXI.Graphics;
  bodyContainer: PIXI.Container;
  background: PIXI.Graphics;
  outputModule: NodeModule;
  resetButton: Button;
  passButton: Button;
  data: number[][];
  backgroundUpdate: () => void;

  constructor(nodes: Nodes, view: PIXI.Container) {
    super();

    this.data = [];

    for (let y = 0; y < 128; y++) {
      const row = new Array(128).fill(0);
      this.data.push(row);
    }

    this.graphic = new PIXI.Graphics();
    this.graphic.beginFill(0x222222);
    this.graphic.drawRect(0, 0, 300, 370);
    this.graphic.endFill();

    this.text = new PIXI.Text("Drawing", {
      fill: 0xffffff,
      fontSize: 35,
    });
    this.text.x = 20;
    this.text.y = 20;

    this.addChild(this.graphic);
    this.addChild(this.text);

    this.bodyContainer = new PIXI.Container();
    this.bodyContainer.x = 22;
    this.bodyContainer.y = 80;

    const width = 256;
    const height = 256;

    const mask = new PIXI.Graphics();
    mask.beginFill(0xffffff);
    mask.drawRect(0, 0, width, height);
    mask.endFill();
    this.bodyContainer.addChild(mask);
    this.bodyContainer.mask = mask;

    this.addChild(this.bodyContainer);

    this.background = new PIXI.Graphics();
    this.background.scale.set(2, 2);

    this.bodyContainer.addChild(this.background);

    this.backgroundUpdate = () => {
      this.background.clear();
      this.background.beginFill(0x000000);
      this.background.drawRect(0, 0, 128, 128);
      this.background.endFill();

      this.background.beginFill(0xffffff);

      for (let y = 0; y < 128; y++) {
        for (let x = 0; x < 128; x++) {
          const z = this.data[y][x];
          this.background.beginFill(new PIXI.Color([z, z, z]));
          this.background.drawRect(x, 128 - y, 1, 1);
        }
      }
      this.background.endFill();
    };
    this.backgroundUpdate();

    this.bodyContainer.eventMode = "static";
    this.background.eventMode = "static";

    let drawing = false;

    const drawPixel = (vector: Vector2, z: number) => {
      const x = Math.floor(vector.x);
      const y = Math.floor(vector.y);

      if (x < 0 || x >= 128 || y < 0 || y >= 128) return;

      const zData = this.data[y][x];

      this.data[y][x] = z + zData > 1 ? 1 : z + zData;
    };
    const draw = (vector: Vector2) => {
      for (let theta = 0; theta < Math.PI * 2; theta += 0.1) {
        for (let r = 0; r < 5; r += 0.5) {
          const currentVector = new Vector2(
            Math.cos(theta) * r,
            Math.sin(theta) * r
          ).add(vector);
          drawPixel(currentVector, 5 - r);
        }
      }
    };

    this.background.on("pointerdown", (event) => {
      drawing = true;
      event.stopPropagation();

      const vector = getMapVectorFromScreen(
        new Vector2(event.x, event.y),
        view
      ).sub(
        new Vector2(
          this.x + this.bodyContainer.x,
          this.y + this.bodyContainer.y
        )
      );
      vector.set(new Vector2(vector.x, height - vector.y).div(2));

      draw(vector);

      this.backgroundUpdate();
    });
    this.background.on("pointermove", (event) => {
      if (!drawing) return;

      const vector = getMapVectorFromScreen(
        new Vector2(event.x, event.y),
        view
      ).sub(
        new Vector2(
          this.x + this.bodyContainer.x,
          this.y + this.bodyContainer.y
        )
      );
      vector.set(new Vector2(vector.x, height - vector.y).div(2));

      draw(vector);

      this.backgroundUpdate();
    });
    this.background.on("pointerleave", () => {
      drawing = false;
    });
    this.background.on("pointerup", () => {
      drawing = false;
    });

    this.resetButton = new Button(
      new Vector2(30, 30),
      0x333333,
      0xffffff,
      "🗑️",
      20
    );
    this.passButton = new Button(
      new Vector2(100, 50),
      0x333333,
      0xffffff,
      "Pass",
      35
    );

    this.resetButton.x = 80;
    this.resetButton.y = 370 - 30;
    this.passButton.x = 300 - 100;
    this.passButton.y = 0;

    this.addChild(this.resetButton);
    this.addChild(this.passButton);

    this.resetButton.on("click", () => {
      this.data = [];

      for (let y = 0; y < 128; y++) {
        const row = new Array(128).fill(0);
        this.data.push(row);
      }
      this.backgroundUpdate();
    });
    this.passButton.on("click", () => {
      const tensor = tf.tensor(this.data).reshape([128, 128, 1]).clone();
      this.outputModule.passData([tensor]);
    });

    this.outputModule = new NodeModule(nodes, "Output tensor");

    this.nodeModules.push(this.outputModule);

    this.outputModule.x = 300 / 2 + 100;
    this.outputModule.y = 370;

    this.outputModule.setGlobalVector(new Vector2(this.x, this.y));

    this.addChild(this.outputModule);
  }
}
class NumberSelectModule extends Module {
  text: PIXI.Text;
  graphic: PIXI.Graphics;
  outputModule: NodeModule;
  passButton: Button;
  buttons: SelectButton[];

  constructor(nodes: Nodes) {
    super();

    this.graphic = new PIXI.Graphics();
    this.graphic.beginFill(0x222222);
    this.graphic.drawRect(0, 0, 230, 200);
    this.graphic.endFill();

    this.text = new PIXI.Text("Select", {
      fill: 0xffffff,
      fontSize: 35,
    });
    this.text.x = 20;
    this.text.y = 20;

    this.addChild(this.graphic);
    this.addChild(this.text);

    this.passButton = new Button(
      new Vector2(100, 50),
      0x333333,
      0xffffff,
      "Pass",
      35
    );
    this.passButton.x = 230 - 100;
    this.passButton.y = 0;

    this.buttons = [];
    for (let i = 0; i < 10; i++) {
      const button = new SelectButton(
        new Vector2(25, 25),
        0x333333,
        0xffffff,
        0xff5555,
        i.toString(),
        25
      );
      button.x = 10 + (i % 5) * 35;
      button.y = 80 + Math.floor(i / 5) * 35;
      this.buttons.push(button);
      this.addChild(button);

      button.on("click", () => {
        this.buttons.forEach((b) => b.unselect());
        button.select();
      });
    }

    this.passButton.on("click", () => {
      const i = this.buttons.findIndex((b) => b.selected);

      if (i == -1) return;

      const result = new Array(10).fill(0);
      result[i] = 1;

      this.outputModule.passData([tf.tensor(result)]);
    });
    this.addChild(this.passButton);

    this.outputModule = new NodeModule(nodes, "Output tensor");

    this.nodeModules.push(this.outputModule);

    this.outputModule.x = 230 / 2 + 50;
    this.outputModule.y = 200;

    this.outputModule.setGlobalVector(new Vector2(this.x, this.y));

    this.addChild(this.outputModule);
  }
}
class ArgmaxModule extends Module {
  text: PIXI.Text;
  graphic: PIXI.Graphics;
  inputModule: NodeModule;
  outputModule: NodeModule;

  constructor(nodes: Nodes) {
    super();

    this.graphic = new PIXI.Graphics();
    this.graphic.beginFill(0x222222);
    this.graphic.drawRect(0, 0, 200, 200);
    this.graphic.endFill();

    this.text = new PIXI.Text("Argmax", {
      fill: 0xffffff,
      fontSize: 35,
    });
    this.text.x = 20;
    this.text.y = 20;

    this.addChild(this.graphic);
    this.addChild(this.text);

    this.inputModule = new NodeModule(nodes, "Input tensor");
    this.outputModule = new NodeModule(nodes, "Output");

    this.inputModule.x = 0;
    this.inputModule.y = 200 / 2;

    this.outputModule.x = 200;
    this.outputModule.y = 200 / 2;

    this.nodeModules.push(this.inputModule);
    this.nodeModules.push(this.outputModule);

    this.inputModule.setGlobalVector(new Vector2(this.x, this.y));
    this.outputModule.setGlobalVector(new Vector2(this.x, this.y));

    this.inputModule.onDataPassed = (data, passCount) => {
      if (!Array.isArray(data) || data.every((t) => !(t instanceof tf.Tensor)))
        return;

      this.outputModule.passData(
        data.map((t: tf.Tensor) => {
          const result = t.argMax(-1).arraySync();
          if (typeof result != "number") return;

          return result;
        }),
        passCount
      );
    };

    this.addChild(this.inputModule);
    this.addChild(this.outputModule);
  }
}
class Printmodule extends Module {
  text: PIXI.Text;
  display: PIXI.Text;
  graphic: PIXI.Graphics;
  inputModule: NodeModule;

  constructor(nodes: Nodes) {
    super();

    this.graphic = new PIXI.Graphics();
    this.graphic.beginFill(0x222222);
    this.graphic.drawRect(0, 0, 200, 200);
    this.graphic.endFill();

    this.text = new PIXI.Text("Print", {
      fill: 0xffffff,
      fontSize: 35,
    });
    this.text.x = 20;
    this.text.y = 20;

    this.display = new PIXI.Text("", {
      fill: 0xffffff,
      fontSize: 35,
      align: "center",
    });
    this.display.anchor.set(0.5, 0.5);

    this.display.x = 200 / 2;
    this.display.y = 200 / 2 + 20;

    this.addChild(this.graphic);
    this.addChild(this.text);
    this.addChild(this.display);

    this.inputModule = new NodeModule(nodes, "Input tensor");

    this.inputModule.x = 200 / 2 + 80;
    this.inputModule.y = 0;

    this.nodeModules.push(this.inputModule);

    this.inputModule.setGlobalVector(new Vector2(this.x, this.y));

    this.addChild(this.inputModule);

    this.inputModule.onDataPassed = (data) => {
      if (!Array.isArray(data) || data.length == 0) return;

      data = data[data.length - 1];
      if (typeof data != "number" && typeof data != "string") return;

      this.display.text = String(data);
    };
  }
}

export {
  Module,
  CoreModule,
  PointInputModule,
  PointZInputModule,
  PointOutputModule,
  PointZOutputModule,
  RangeInputModule,
  TensorPackModule,
  TensorUnpackModule,
  DrawingModule,
  NumberSelectModule,
  ArgmaxModule,
  Printmodule,
};
