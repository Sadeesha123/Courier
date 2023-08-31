import { useState } from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPlus, faStarHalfStroke } from "@fortawesome/free-solid-svg-icons";
import Sidepanel from "../../components/sidepanel";

function PackageStatus() {
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
      <div className="main-body-container h-screen w-full flex flex-row">
        <Sidepanel />
        <div className="w-5/6 side-panel bg-slate-100 p-5">
          <div className="common-body p-5 flex flex-col h-full items-center justify-center">
            <div className="form-body w-[80%] flex flex-col p-5 mx-auto items-center justify-center bg-white rounded-lg shadow-md shadow-slate-300">
              <form onSubmit={handleSubmit} className="flex flex-col w-full">
                <label
                  htmlFor="orderId"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Send From
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
                  Send to
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
                  className="mb-4 p-2 rounded-lg border border-gray-300"
                  accept="image/*"
                  multiple
                  required
                  onChange={handleImageUpload}
                />

                <div className="flex flex-row space-x-3 w-full">
                  <Link to="/CustomerHome" className="w-full"><button
                    type="submit"
                    className="py-2 px-4 bg-blue-500 text-white rounded-lg w-full"
                  >
                    Approve
                  </button></Link>
                  <Link to="/Inquiry" className="w-full"><button
                    type="submit"
                    className="py-2 px-4 border-[1px] border-blue-500 text-gray-900 rounded-lg w-full"
                  >
                    Send for Inquiry
                  </button></Link>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PackageStatus;
