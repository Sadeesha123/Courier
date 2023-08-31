import { useState } from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPlus, faStarHalfStroke } from "@fortawesome/free-solid-svg-icons";
import Sidepanel from "../../components/sidepanel";
import bg from '../../images/mainbg1.jpg';

function Inquiry() {
  const handleSubmit = (event) => {
    event.preventDefault();
  };

  function handleImageUpload(event) {
    const selectedFiles = event.target.files;
    const fileCount = selectedFiles.length;

    if (fileCount < 1 || fileCount > 5) {
      // Display an error message or handle the validation error accordingly
      console.log("Please select between 1 and 5 images.");
      return;
    }
  }

  return (
    <div className="main-body h-screen w-full bg-slate-100">
      <img src={bg} alt="" srcset="" className="object-cover w-[100%] h-[100%] fixed" />
      <div className="main-body-container w-full flex flex-row absolute">
        <Sidepanel />
        <div className="w-5/6 side-panel p-5 md:ml-[300px] ml-16">
          <div className="common-body p-5 flex flex-col h-full items-center">
            <div className="form-body md:w-[80%] w-full flex flex-col p-5 mx-auto items-center justify-center bg-white rounded-lg">
              <form onSubmit={handleSubmit} className="flex flex-col w-full">
                <label
                  htmlFor="orderId"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Package No
                </label>
                <input
                  type="text"
                  id="orderId"
                  name="orderId"
                  className="mb-4 p-2 rounded-lg border border-gray-300"
                  required
                />

                <label
                  htmlFor="pickup"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Reciever 
                </label>
                <input
                  type="text"
                  id="pickup"
                  name="pickup"
                  className="mb-4 p-2 rounded-lg border border-gray-300"
                  required
                />

                <label
                  htmlFor="destination"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Package Description
                </label>
                <input
                  type="text"
                  id="destination"
                  name="destination"
                  className="mb-4 p-2 rounded-lg border border-gray-300"
                  required
                />

                <label
                  htmlFor="images"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Upload Images (1-5)
                </label>
                <input
                  type="file"
                  id="images"
                  name="images"
                  className="mb-4 py-2 pl-5 file:rounded-lg rounded-lg border border-gray-300"
                  accept="image/*"
                  multiple
                  onChange={handleImageUpload}
                  required
                />

                <button
                  type="submit"
                  className="py-2 px-4 bg-blue-500 text-white rounded-lg w-52 ml-auto mt-8"
                >
                  Send for Inquiry
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Inquiry;
