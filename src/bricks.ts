import * as PIXI from "pixi.js";
import * as tf from "@tensorflow/tfjs";
import { Vector2, getMapVectorFromScreen } from "./math";
import { ActivationIdentifier } from "@tensorflow/tfjs-layers/dist/keras_format/activation_config";
import { InitializerIdentifier } from "@tensorflow/tfjs-layers/dist/initializers";

class Bricks extends PIXI.Container {
  bricks: Brick[];
  currentZindex: number;

  constructor() {
    super();
    this.bricks = [];
    this.sortableChildren = true;
    this.currentZindex = 0;
  }
  addBrick(brick: Brick) {
    this.bricks.push(brick);
    this.addChild(brick);
  }
  getIndex(x: number) {
    let i = 0;
    for (i = 0; i < this.bricks.length; i++) {
      const brick = this.bricks[i];
      x -= brick.size.x;

      if (x < 0) return i;
    }
    return i - 1;
  }
  get(i: number) {
    return this.bricks[i >= this.bricks.length ? this.bricks.length - 1 : i];
  }
  move(i: number, brick: Brick) {
    const currIndex = this.bricks.findIndex((b) => b == brick);
    this.bricks.splice(currIndex, 1);
    this.bricks = this.bricks
      .slice(0, i)
      .concat(brick, ...this.bricks.slice(i));
    this.sort();
  }
  sort() {
    let x = 0;
    for (let i = 0; i < this.bricks.length; i++) {
      const brick = this.bricks[i];
      brick.x = x;
      brick.y = -brick.size.y / 2;
      x += brick.size.x;
    }
  }
}

interface IOutputConfig {
  config: tf.ModelCompileArgs;
  epochs: number;
}
class Brick extends PIXI.Container {
  size: Vector2;
  graphic: PIXI.Graphics;
  text: PIXI.Text;
  dragging: boolean;
  onDoubleClick: () => void;
  toLayer: () => tf.layers.Layer | IOutputConfig | null;

  constructor(
    size: Vector2,
    text: string,
    color: PIXI.ColorSource,
    bricks: Bricks
  ) {
    super();
    this.onDoubleClick = () => {};
    this.toLayer = () => tf.layers.flatten();

    this.dragging = false;

    this.size = size;
    this.graphic = new PIXI.Graphics();

    this.graphic.beginFill(color);
    this.graphic.drawRect(0, 0, size.x, size.y);
    this.graphic.endFill();
    this.graphic.eventMode = "static";

    this.text = new PIXI.Text(text, {
      wordWrapWidth: size.x,
      wordWrap: true,
      fill: 0xffffff,
      align: "center",
      fontSize: 35,
    });
    this.text.anchor.set(0.5, 0.5);
    this.text.x = size.x / 2;
    this.text.y = size.y / 2;
    this.addChild(this.graphic);
    this.addChild(this.text);
    this.zIndex = bricks.currentZindex;
    this.pivot.set(0, 0.5);

    this.eventMode = "static";

    let lastClick: number | null = null;
    this.on("click", (event) => {
      if (event.button == 0) {
        if (lastClick != null && Date.now() - lastClick < 200) {
          this.onDoubleClick();
        } else {
          lastClick = Date.now();
        }
      }
    });
    this.on("pointerdown", (event) => {
      if (event.button == 2) {
        const index = bricks.bricks.findIndex((v) => v == this);
        if (index != -1) {
          bricks.bricks.splice(index, 1);
          bricks.removeChild(this);
          bricks.sort();
        }
      }
    });
  }
  registerEvents(
    app: PIXI.Application<PIXI.ICanvas>,
    view: PIXI.Container<PIXI.DisplayObject>,
    bricks: Bricks
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
      this.zIndex = ++bricks.currentZindex;

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
    };
    const onPointerUp = () => {
      this.dragging = false;

      const i = bricks.getIndex(this.x - bricks.x);
      bricks.move(i, this);

      app.stage.off("pointermove", onPointerMove);
      app.stage.off("pointerup", onPointerUp);
      app.stage.off("pointerleave", onPointerUp);
    };
  }
}
interface IModalItem {
  name: string;
  tag: string;
  required?: boolean;
  default?: string | number;
  options?: string[];
  type: "string" | "number";
}

function newModal(items: IModalItem[], onSubmit: (data: FormData) => boolean) {
  const modal = document.getElementById("modal") as HTMLDivElement;

  if (!modal) return null;

  modal.innerHTML = "";

  const form = document.createElement("form");

  items.forEach((item) => {
    const label = document.createElement("label");

    label.textContent = item.name + " ";

    if (item.options == undefined) {
      const input = document.createElement("input");

      input.type = item.type == "string" ? "text" : "number";
      input.autocomplete = "off";
      input.spellcheck = false;
      input.name = item.tag;
      if (item.default != undefined) {
        input.defaultValue = String(item.default);
      }
      input.required = item.required || false;

      label.appendChild(input);
    } else {
      const select = document.createElement("select");

      select.name = item.tag;

      item.options.forEach((o) => {
        const option = document.createElement("option");
        option.value = o;
        option.textContent = o;
        select.appendChild(option);
      });
      select.value = String(item.default);

      label.appendChild(select);
    }

    form.appendChild(label);
  });

  const button = document.createElement("input");
  button.id = "submit";
  button.type = "submit";
  button.value = "Done";
  form.appendChild(button);

  modal.appendChild(form);

  modal.hidden = false;

  form.onsubmit = (event) => {
    event.preventDefault();

    const formData = new FormData(form);

    try {
      if (onSubmit(formData)) {
        modal.hidden = true;
        form.onsubmit = null;
      }
    } catch (err) {
      if (err instanceof Error) {
        alert(err.message);
      }
    }
  };

  return form;
}

class Dense extends Brick {
  units: number;
  activation?: ActivationIdentifier;
  kernelInitializer?: InitializerIdentifier;

  constructor(
    bricks: Bricks,
    size?: Vector2,
    text?: string,
    color?: PIXI.ColorSource
  ) {
    super(
      size || new Vector2(250, 250),
      text || "Dense",
      color || 0x11aa55,
      bricks
    );

    this.units = 64;
    this.activation = "relu";
    this.kernelInitializer = "none";

    this.onDoubleClick = () =>
      newModal(
        [
          {
            name: "Units",
            tag: "units",
            required: true,
            type: "number",
            default: this.units,
          },
          {
            name: "Activation",
            tag: "activation",
            required: false,
            type: "string",
            options: ["none", "relu", "sigmoid", "softmax"],
            default: this.activation || "none",
          },
          {
            name: "Kernel Initializer",
            tag: "kernelInitializer",
            required: false,
            type: "string",
            options: ["none", "varianceScaling"],
            default: this.kernelInitializer || "none",
          },
        ],
        (data) => {
          const units = Number(data.get("units"));
          const activation = data.get("activation") as string;
          const kernelInitializer = data.get("kernelInitializer") as string;

          this.units = units;
          this.activation =
            activation != "none"
              ? (activation as ActivationIdentifier)
              : undefined;
          this.kernelInitializer =
            kernelInitializer != "none"
              ? (kernelInitializer as InitializerIdentifier)
              : undefined;

          return true;
        }
      );
    this.toLayer = () => {
      return tf.layers.dense({
        units: this.units,
        activation: this.activation,
      });
    };
  }
}
class Input extends Dense {
  inputShape: number[];

  constructor(bricks: Bricks) {
    super(bricks, new Vector2(300, 300), "Input", 0xaa1111);
    this.inputShape = [1];

    this.onDoubleClick = () =>
      newModal(
        [
          {
            name: "Input shape",
            tag: "shape",
            required: true,
            type: "string",
            default: JSON.stringify(this.inputShape),
          },
          {
            name: "Units",
            tag: "units",
            required: true,
            type: "number",
            default: this.units,
          },
          {
            name: "Activation",
            tag: "activation",
            required: false,
            type: "string",
            options: ["none", "relu", "sigmoid", "softmax"],
            default: this.activation || "none",
          },
        ],
        (data) => {
          const shapeString = data.get("shape") as string;
          const units = Number(data.get("units"));
          const activation = data.get("activation") as string;

          const shape = JSON.parse(shapeString);

          this.inputShape = shape;
          this.units = units;

          this.activation =
            activation != "none"
              ? (activation as ActivationIdentifier)
              : undefined;

          return true;
        }
      );
    this.toLayer = () => {
      return tf.layers.dense({
        inputShape: this.inputShape,
        units: this.units,
        activation: this.activation,
      });
    };
  }
}
class Conv2D extends Brick {
  kernelSize: number;
  _filters: number;
  strides: number;
  kernelInitializer?: InitializerIdentifier;
  activation?: ActivationIdentifier;

  constructor(
    bricks: Bricks,
    size?: Vector2,
    text?: string,
    color?: PIXI.ColorSource
  ) {
    super(
      size || new Vector2(200, 200),
      text || "Conv2D",
      color || 0x115511,
      bricks
    );
    this.kernelSize = 5;
    this._filters = 8;
    this.strides = 1;
    this.kernelInitializer = "varianceScaling";

    this.onDoubleClick = () =>
      newModal(
        [
          {
            name: "Kernel size",
            tag: "kernelsize",
            required: true,
            type: "number",
            default: this.kernelSize,
          },
          {
            name: "Filters",
            tag: "filters",
            required: true,
            type: "number",
            default: this._filters,
          },
          {
            name: "Strides",
            tag: "strides",
            type: "number",
            default: this.strides,
          },
          {
            name: "Activation",
            tag: "activation",
            required: false,
            type: "string",
            options: ["none", "relu", "sigmoid", "softmax"],
            default: this.activation || "none",
          },
          {
            name: "Kernel Initializer",
            tag: "kernelInitializer",
            required: false,
            type: "string",
            options: ["none", "varianceScaling"],
            default: this.kernelInitializer || "none",
          },
        ],
        (data) => {
          const kernelSize = Number(data.get("kernelsize"));
          const filters = Number(data.get("filters"));
          const strides = Number(data.get("strides"));
          const activation = data.get("activation") as string;
          const kernelInitializer = data.get("kernelInitializer") as string;

          this.kernelSize = kernelSize;
          this._filters = filters;
          this.strides = strides;

          this.activation =
            activation != "none"
              ? (activation as ActivationIdentifier)
              : undefined;
          this.kernelInitializer =
            kernelInitializer != "none"
              ? (kernelInitializer as InitializerIdentifier)
              : undefined;

          return true;
        }
      );
    this.toLayer = () => {
      return tf.layers.conv2d({
        kernelSize: this.kernelSize,
        filters: this._filters,
        strides: this.strides,
        activation: this.activation,
        kernelInitializer: this.kernelInitializer,
      });
    };
  }
}
class ImageInput extends Conv2D {
  inputShape: number[];
  constructor(bricks: Bricks) {
    super(bricks, new Vector2(300, 300), "Image Input", 0x551111);

    this.inputShape = [128, 128, 1];

    this.onDoubleClick = () =>
      newModal(
        [
          {
            name: "Input shape",
            tag: "shape",
            required: true,
            type: "string",
            default: JSON.stringify(this.inputShape),
          },
          {
            name: "Kernel size",
            tag: "kernelsize",
            required: true,
            type: "number",
            default: this.kernelSize,
          },
          {
            name: "Filters",
            tag: "filters",
            required: true,
            type: "number",
            default: this._filters,
          },
          {
            name: "Strides",
            tag: "strides",
            type: "number",
            default: this.strides,
          },
          {
            name: "Activation",
            tag: "activation",
            required: false,
            type: "string",
            options: ["none", "relu", "sigmoid", "softmax"],
            default: this.activation || "none",
          },
          {
            name: "Kernel Initializer",
            tag: "kernelInitializer",
            required: false,
            type: "string",
            options: ["none", "varianceScaling"],
            default: this.kernelInitializer || "none",
          },
        ],
        (data) => {
          const kernelSize = Number(data.get("kernelsize"));
          const filters = Number(data.get("filters"));
          const strides = Number(data.get("strides"));
          const activation = data.get("activation") as string;
          const kernelInitializer = data.get("kernelInitializer") as string;
          const inputShape = JSON.parse(data.get("shape") as string);

          this.inputShape = inputShape;
          this.kernelSize = kernelSize;
          this._filters = filters;
          this.strides = strides;

          this.activation =
            activation != "none"
              ? (activation as ActivationIdentifier)
              : undefined;
          this.kernelInitializer =
            kernelInitializer != "none"
              ? (kernelInitializer as InitializerIdentifier)
              : undefined;

          return true;
        }
      );
    this.toLayer = () => {
      return tf.layers.conv2d({
        inputShape: this.inputShape,
        kernelSize: this.kernelSize,
        filters: this._filters,
        strides: this.strides,
        activation: this.activation,
        kernelInitializer: this.kernelInitializer,
      });
    };
  }
}
class Pooling extends Brick {
  poolSize: [number, number];
  strides: [number, number];

  constructor(bricks: Bricks) {
    super(new Vector2(200, 200), "Pooling", 0x119911, bricks);

    this.poolSize = [2, 2];
    this.strides = [2, 2];

    this.onDoubleClick = () =>
      newModal(
        [
          {
            name: "Pool size",
            tag: "poolsize",
            required: true,
            type: "string",
            default: JSON.stringify(this.poolSize),
          },
          {
            name: "Strides",
            tag: "strides",
            required: true,
            type: "string",
            default: JSON.stringify(this.strides),
          },
        ],
        (data) => {
          const poolSize = JSON.parse(data.get("poolsize") as string);
          const strides = JSON.parse(data.get("strides") as string);

          this.poolSize = poolSize;
          this.strides = strides;

          return true;
        }
      );
    this.toLayer = () => {
      return tf.layers.maxPool2d({
        poolSize: this.poolSize,
        strides: this.strides,
      });
    };
  }
}
class Flatten extends Brick {
  constructor(bricks: Bricks) {
    super(new Vector2(220, 220), "Flatten", 0x111155, bricks);

    this.toLayer = () => {
      return tf.layers.flatten();
    };
  }
}
class Output extends Brick {
  loss: string;
  epochs: number;

  constructor(bricks: Bricks) {
    super(new Vector2(300, 300), "Output", 0xaa11aa, bricks);

    this.loss = "meanSquaredError";
    this.epochs = 500;

    this.onDoubleClick = () =>
      newModal(
        [
          {
            name: "Loss",
            tag: "loss",
            required: true,
            type: "string",
            options: ["meanSquaredError", "categoricalCrossentropy"],
            default: this.loss,
          },
          {
            name: "Epochs",
            tag: "epochs",
            required: true,
            type: "number",
            default: this.epochs,
          },
        ],
        (data) => {
          const loss = data.get("loss") as string;
          const epochs = Number(data.get("epochs"));

          this.loss = loss;
          this.epochs = epochs;

          return true;
        }
      );
    this.toLayer = () => {
      return {
        epochs: this.epochs,
        config: {
          optimizer: tf.train.adam(),
          loss: "meanSquaredError",
          metrics: ["accuracy"],
        },
      };
    };
  }
}

export {
  Bricks,
  Brick,
  Input,
  Dense,
  Conv2D,
  ImageInput,
  Pooling,
  Flatten,
  Output,
  newModal,
  IOutputConfig,
};
