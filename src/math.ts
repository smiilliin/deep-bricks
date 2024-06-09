import * as PIXI from "pixi.js";

class Vector2 {
  x: number;
  y: number;

  constructor(x: number, y: number) {
    this.x = x;
    this.y = y;
  }
  add(vector: Vector2) {
    return new Vector2(this.x + vector.x, this.y + vector.y);
  }
  sub(vector: Vector2) {
    return new Vector2(this.x - vector.x, this.y - vector.y);
  }
  mul(x: number) {
    return new Vector2(this.x * x, this.y * x);
  }
  div(x: number) {
    return new Vector2(this.x / x, this.y / x);
  }
  clone() {
    return new Vector2(this.x, this.y);
  }
  compare(vector: Vector2) {
    return this.x == vector.x && this.y == vector.y;
  }
  distance(vector: Vector2) {
    return Math.sqrt(
      Math.pow(vector.x - this.x, 2) + Math.pow(vector.y - this.y, 2)
    );
  }
  set(vector: Vector2) {
    this.x = vector.x;
    this.y = vector.y;
  }
  floor() {
    return new Vector2(Math.floor(this.x), Math.floor(this.y));
  }
  ceil() {
    return new Vector2(Math.ceil(this.x), Math.ceil(this.y));
  }
  toString(): string {
    return `(${this.x}, ${this.y})`;
  }
}

const getMapVectorFromScreen = (vector: Vector2, view: PIXI.Container) => {
  return new Vector2(
    (vector.x - view.x) / view.scale.x,
    (vector.y - view.y) / view.scale.y
  );
};
const getScreenVectorFromMap = (vector: Vector2, view: PIXI.Container) => {
  return new Vector2(
    vector.x * view.scale.x + view.x,
    vector.y * view.scale.y + view.y
  );
};

// type Tensor = number[];

export { Vector2, getMapVectorFromScreen, getScreenVectorFromMap };
